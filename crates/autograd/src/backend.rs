//! Backend abstraction for heavy ops. Today: matmul forward only.
//!
//! Transformer training is ~90% matmul FLOPs; moving matmul to GPU swings the
//! big lever without requiring device-resident tensors. Host `Vec<f32>`
//! stays authoritative; GPU backends upload, compute, and download per
//! call. Non-matmul ops (softmax, elementwise, norm, gather) stay on CPU.
//!
//! The trait is additive — future ops land as new methods with CPU
//! fallbacks so a backend does not need to implement every op day one.

use crate::{AutogradError, Result};
#[cfg(any(feature = "metal", feature = "cuda"))]
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Metal,
    Cuda,
}

pub type CausalSdpaHostGradTriplet = (Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>);

pub use cuda_kernels::ring_attention::RingBlockDims;

#[derive(Debug, Clone, Copy)]
pub struct CausalSdpaDeviceBackwardArgs<'a> {
    pub q: &'a DeviceHandle,
    pub k: &'a DeviceHandle,
    pub v: &'a DeviceHandle,
    pub upstream: &'a DeviceHandle,
    pub shape: &'a [usize],
    pub need_grad_q: bool,
    pub need_grad_k: bool,
    pub need_grad_v: bool,
}

pub type CausalSdpaDeviceGradTriplet = (
    Option<DeviceHandle>,
    Option<DeviceHandle>,
    Option<DeviceHandle>,
);

#[cfg(feature = "metal")]
#[derive(Debug, Clone)]
pub struct MlxHandle {
    inner: Arc<MlxHandleInner>,
}

#[cfg(feature = "metal")]
#[derive(Debug)]
struct MlxHandleInner {
    ptr: *mut mlx_sys::mlx_array,
}

#[cfg(feature = "metal")]
// Safety: `MlxHandleInner` is just an opaque MLX array pointer. All MLX FFI
// access in this crate is serialized through `mlx_sys::mlx_guard()`, so moving
// or sharing the pointer wrapper across threads does not introduce
// unsynchronized MLX calls.
unsafe impl Send for MlxHandleInner {}

#[cfg(feature = "metal")]
// Safety: see the `Send` impl above. Shared access only hands the opaque
// pointer back to MLX while holding `mlx_guard()`.
unsafe impl Sync for MlxHandleInner {}

#[cfg(feature = "metal")]
impl MlxHandle {
    pub(crate) fn from_raw(ptr: *mut mlx_sys::mlx_array) -> Self {
        Self {
            inner: Arc::new(MlxHandleInner { ptr }),
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut mlx_sys::mlx_array {
        self.inner.ptr
    }
}

#[cfg(feature = "metal")]
impl Drop for MlxHandleInner {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        let _guard = crate::backend_metal::mlx_guard();

        // Safety: `ptr` is owned by this handle, came from MLX FFI allocation,
        // and this Drop impl is the unique free path for the wrapped array.
        // `mlx_guard()` serializes the free against all other guarded MLX FFI calls.
        unsafe {
            mlx_sys::mlx_array_free(self.ptr);
        }
    }
}

#[cfg(feature = "metal")]
// Safety: `MlxHandle` owns an MLX array pointer. MLX's global stream is not
// safe for concurrent mutation, but all MLX FFI use in this crate is
// serialized by `mlx_sys::mlx_guard()`, which is the synchronization
// boundary for moving these opaque handles across threads.
unsafe impl Send for MlxHandle {}

#[cfg(feature = "metal")]
// Safety: see the `Send` impl above. Shared references are only used to pass
// opaque handles into MLX while holding `mlx_guard()`.
unsafe impl Sync for MlxHandle {}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "no-cuda", allow(dead_code))]
pub struct CudaStorage {
    inner: Arc<cudarc::driver::CudaSlice<f32>>,
}

#[cfg(feature = "cuda")]
#[cfg_attr(feature = "no-cuda", allow(dead_code))]
impl CudaStorage {
    pub(crate) fn new(inner: cudarc::driver::CudaSlice<f32>) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    pub(crate) fn slice(&self) -> &cudarc::driver::CudaSlice<f32> {
        self.inner.as_ref()
    }

    /// Strong-count of the backing device buffer. `1` means this handle is the
    /// sole owner, so an in-place mutation cannot corrupt a sibling that shares
    /// the same `Arc` (grads fan out by refcount clone, not deep copy — see
    /// `clone_tensor`). Used to gate in-place gradient accumulation.
    pub(crate) fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "no-cuda", allow(dead_code))]
pub struct CudaBf16Storage {
    inner: Arc<cudarc::driver::CudaSlice<u16>>,
    /// `true` when `inner` is a NON-OWNING view over device memory owned by a
    /// foreign allocator (the infer-cuda rollout engine's resident BF16 base
    /// `DeviceMatrix`, shared after a LoRA re-merge). The backing bytes belong
    /// to the other engine; this handle must NEVER free them. The custom `Drop`
    /// below leaks the inner `Arc` so the foreign allocation survives this
    /// handle. Default `false` (owned) keeps the existing path byte-identical.
    borrowed: bool,
}

#[cfg(feature = "cuda")]
impl Drop for CudaBf16Storage {
    fn drop(&mut self) {
        if !self.borrowed {
            return;
        }
        // Foreign-borrowed view: leak the inner `Arc` so the foreign allocation
        // survives this handle (the infer engine frees the bytes at its own
        // teardown). See `CudaFp8BlockScaledStorage::drop` for the full rationale.
        std::mem::forget(self.inner.clone());
    }
}

#[cfg(feature = "cuda")]
#[cfg_attr(feature = "no-cuda", allow(dead_code))]
impl CudaBf16Storage {
    pub(crate) fn new(inner: cudarc::driver::CudaSlice<u16>) -> Self {
        Self {
            inner: Arc::new(inner),
            borrowed: false,
        }
    }

    /// Construct a NON-OWNING BF16 view over a device buffer owned by a foreign
    /// allocator. The caller guarantees `inner` was built from a foreign device
    /// pointer (e.g. via `CudaStream::upgrade_device_ptr`) in the same primary
    /// context, and that the foreign owner keeps it resident for the lifetime
    /// of this handle. On drop the inner `Arc` is leaked (never freed). Used by
    /// the post-LoRA-merge frozen-base refresh path.
    pub(crate) fn new_borrowed(inner: cudarc::driver::CudaSlice<u16>) -> Self {
        Self {
            inner: Arc::new(inner),
            borrowed: true,
        }
    }

    pub(crate) fn slice(&self) -> &cudarc::driver::CudaSlice<u16> {
        self.inner.as_ref()
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "no-cuda", allow(dead_code))]
pub struct CudaFp8BlockScaledStorage {
    weight: Arc<cudarc::driver::CudaSlice<u8>>,
    scales: Arc<cudarc::driver::CudaSlice<f32>>,
    rows: usize,
    cols: usize,
    block_m: usize,
    block_k: usize,
    /// `true` when `weight`/`scales` are NON-OWNING views over device memory
    /// owned by a *foreign* allocator (the infer-cuda rollout/eval engine's
    /// resident FP8 base `DeviceMatrix`, shared via `--share-frozen-base`). The
    /// backing bytes belong to the other engine; this handle must NEVER free
    /// them. The custom `Drop` below leaks the inner `Arc`s so the foreign
    /// allocation survives this handle (the infer engine frees them at its own
    /// teardown). Default `false` (owned) keeps the existing path byte-identical.
    borrowed: bool,
}

#[cfg(feature = "cuda")]
impl Drop for CudaFp8BlockScaledStorage {
    fn drop(&mut self) {
        if !self.borrowed {
            return;
        }
        // Foreign-borrowed view: the inner `CudaSlice`s wrap device pointers
        // OWNED by the infer engine (built via `upgrade_device_ptr`); a
        // `CudaSlice::Drop` would `cuMemFree` the foreign bytes. After this
        // `drop()` returns, Rust's drop-glue still drops `self.weight`/
        // `self.scales` (one `Arc` strong-count decrement each). Bump each
        // strong count and leak the bump so that decrement lands at >= 1 — the
        // inner `CudaSlice` is never dropped here, and the infer engine frees the
        // bytes exactly once when IT drops.
        //
        // NOTE: `ptr::read(&field)` + `forget` does NOT work — it leaks a bitwise
        // *copy* while drop-glue still drops the original field → double-free.
        std::mem::forget(self.weight.clone());
        std::mem::forget(self.scales.clone());
    }
}

#[cfg(feature = "cuda")]
#[cfg_attr(feature = "no-cuda", allow(dead_code))]
impl CudaFp8BlockScaledStorage {
    pub(crate) fn new(
        weight: cudarc::driver::CudaSlice<u8>,
        scales: cudarc::driver::CudaSlice<f32>,
        rows: usize,
        cols: usize,
        block_m: usize,
        block_k: usize,
    ) -> Self {
        Self {
            weight: Arc::new(weight),
            scales: Arc::new(scales),
            rows,
            cols,
            block_m,
            block_k,
            borrowed: false,
        }
    }

    /// Construct a NON-OWNING block-scaled FP8 view over device buffers owned by
    /// a foreign allocator. The caller guarantees `weight`/`scales` were built
    /// from foreign device pointers (e.g. via `CudaStream::upgrade_device_ptr`)
    /// in the same primary context, and that the foreign owner keeps them
    /// resident for the lifetime of this handle. On drop the inner `Arc`s are
    /// leaked (never freed) — see the [`Drop`] impl. Used by the
    /// `--share-frozen-base` train-infer weight-sharing path.
    pub(crate) fn new_borrowed(
        weight: cudarc::driver::CudaSlice<u8>,
        scales: cudarc::driver::CudaSlice<f32>,
        rows: usize,
        cols: usize,
        block_m: usize,
        block_k: usize,
    ) -> Self {
        Self {
            weight: Arc::new(weight),
            scales: Arc::new(scales),
            rows,
            cols,
            block_m,
            block_k,
            borrowed: true,
        }
    }

    pub(crate) fn weight(&self) -> &cudarc::driver::CudaSlice<u8> {
        self.weight.as_ref()
    }

    /// Identity of the backing weight buffer, for the backend's dequant cache.
    pub(crate) fn weight_key(&self) -> usize {
        Arc::as_ptr(&self.weight) as usize
    }

    pub(crate) fn scales(&self) -> &cudarc::driver::CudaSlice<f32> {
        self.scales.as_ref()
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn block_m(&self) -> usize {
        self.block_m
    }

    pub(crate) fn block_k(&self) -> usize {
        self.block_k
    }
}

/// Frozen NVFP4 base weight held at 4 bits. The 27B base is 15.2 GB here
/// against 54 GB as BF16, which is the whole point of a 4-bit frozen base;
/// sm_90 has no FP4 tensor cores, so the forward dequantizes to BF16 scratch
/// per projection and the GEMM rides cuBLAS. Structurally the FP4 twin of
/// [`CudaFp8BlockScaledStorage`], with the group scales in FP8 E4M3 (u8) plus
/// one per-tensor F32 global scale.
#[derive(Debug, Clone)]
#[cfg(feature = "cuda")]
#[cfg_attr(feature = "no-cuda", allow(dead_code))]
pub struct CudaFp4E2M1GroupStorage {
    weight: Arc<cudarc::driver::CudaSlice<u8>>,
    scales: Arc<cudarc::driver::CudaSlice<u8>>,
    /// `None` on the Marlin path: the repack's folded global lives in
    /// `marlin_global` as a plain scalar, so a borrowed view owns no buffer it
    /// would have to leak on drop.
    global_scale: Option<Arc<cudarc::driver::CudaSlice<f32>>>,
    marlin_global: f32,
    rows: usize,
    cols: usize,
    group_size: usize,
    scale_cols: usize,
    /// `true` when `weight`/`scales` carry the Marlin tensor-core layout instead
    /// of the group layout — the form a serving engine keeps after its repack
    /// releases the group bytes, which a shared frozen base has to read as-is.
    /// `global_scale` then holds the repack's bf16 value (2^119 bias folded in).
    marlin: bool,
    /// Same foreign-view semantics as `CudaFp8BlockScaledStorage::borrowed`.
    borrowed: bool,
}

#[cfg(feature = "cuda")]
impl Drop for CudaFp4E2M1GroupStorage {
    fn drop(&mut self) {
        if !self.borrowed {
            return;
        }
        // Foreign-borrowed view: leak one strong count per buffer so drop-glue
        // lands at >= 1 and the infer engine stays the sole owner. See the
        // `CudaFp8BlockScaledStorage` Drop for the full rationale.
        std::mem::forget(self.weight.clone());
        std::mem::forget(self.scales.clone());
        if let Some(global) = self.global_scale.as_ref() {
            std::mem::forget(global.clone());
        }
    }
}

#[cfg(feature = "cuda")]
#[cfg_attr(feature = "no-cuda", allow(dead_code))]
impl CudaFp4E2M1GroupStorage {
    pub(crate) fn new(
        weight: cudarc::driver::CudaSlice<u8>,
        scales: cudarc::driver::CudaSlice<u8>,
        global_scale: cudarc::driver::CudaSlice<f32>,
        rows: usize,
        cols: usize,
        group_size: usize,
        scale_cols: usize,
    ) -> Self {
        Self {
            weight: Arc::new(weight),
            scales: Arc::new(scales),
            global_scale: Some(Arc::new(global_scale)),
            marlin_global: 0.0,
            rows,
            cols,
            group_size,
            scale_cols,
            marlin: false,
            borrowed: false,
        }
    }

    /// Non-owning Marlin view over a serving engine's resident base. Same
    /// foreign-ownership contract as `CudaFp8BlockScaledStorage::new_borrowed`.
    pub(crate) fn new_borrowed_marlin(
        weight: cudarc::driver::CudaSlice<u8>,
        scales: cudarc::driver::CudaSlice<u8>,
        global_scale: f32,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self {
            weight: Arc::new(weight),
            scales: Arc::new(scales),
            global_scale: None,
            marlin_global: global_scale,
            rows,
            cols,
            group_size: 16,
            scale_cols: cols / 16,
            marlin: true,
            borrowed: true,
        }
    }

    pub(crate) fn marlin_global(&self) -> f32 {
        self.marlin_global
    }

    pub(crate) fn is_marlin(&self) -> bool {
        self.marlin
    }

    pub(crate) fn weight(&self) -> &cudarc::driver::CudaSlice<u8> {
        self.weight.as_ref()
    }

    /// Identity of the backing weight buffer, for the backend's dequant cache.
    pub(crate) fn weight_key(&self) -> usize {
        Arc::as_ptr(&self.weight) as usize
    }

    pub(crate) fn scales(&self) -> &cudarc::driver::CudaSlice<u8> {
        self.scales.as_ref()
    }

    pub(crate) fn global_scale(&self) -> Option<&cudarc::driver::CudaSlice<f32>> {
        self.global_scale.as_deref()
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn group_size(&self) -> usize {
        self.group_size
    }

    pub(crate) fn scale_cols(&self) -> usize {
        self.scale_cols
    }
}

#[derive(Debug, Clone)]
pub enum DeviceHandle {
    Cpu(Vec<f32>),
    #[cfg(feature = "metal")]
    Metal(MlxHandle),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
    #[cfg(feature = "cuda")]
    CudaBf16(CudaBf16Storage),
    #[cfg(feature = "cuda")]
    CudaFp8BlockScaled(CudaFp8BlockScaledStorage),
    #[cfg(feature = "cuda")]
    CudaFp4E2M1Group(CudaFp4E2M1GroupStorage),
}

impl DeviceHandle {
    /// Strong-count of the backing device buffer when it lives on a refcounted
    /// device allocation (CUDA f32). `Some(1)` means this handle is the sole
    /// owner and an in-place mutation is safe; `Some(n>1)` means a sibling
    /// aliases the same buffer (grads fan out by `Arc` clone — see
    /// `clone_tensor`) so in-place would corrupt it. `None` for handles with no
    /// meaningful single-owner semantics here (CPU/Metal/bf16/fp8), which the
    /// caller treats as "not provably unique" → allocating fallback.
    pub fn device_buffer_strong_count(&self) -> Option<usize> {
        match self {
            #[cfg(feature = "cuda")]
            DeviceHandle::Cuda(storage) => Some(storage.strong_count()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceGradClipResult {
    pub pre_clip_norm: f64,
    pub clipped_grads: Option<Vec<DeviceHandle>>,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearAttentionScanBackwardParams {
    pub batch: usize,
    pub seq_len: usize,
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub eps: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearAttentionScanBackwardArgs<'a> {
    pub params: LinearAttentionScanBackwardParams,
    pub upstream: &'a [f32],
    pub z: &'a [f32],
    pub a_proj: &'a [f32],
    pub dt_bias: &'a [f32],
    pub a_log: &'a [f32],
    pub norm_weight: &'a [f32],
    pub preact: &'a [f32],
    pub beta: &'a [f32],
    pub exp_g: &'a [f32],
    pub kv_mem: &'a [f32],
    pub state_history: &'a [f32],
    pub final_state: &'a [f32],
}

#[derive(Debug, Clone)]
pub struct LinearAttentionScanBackwardGrads {
    pub dqkv: Vec<f32>,
    pub dz: Vec<f32>,
    pub db: Vec<f32>,
    pub da: Vec<f32>,
    pub ddt: Vec<f32>,
    pub da_log: Vec<f32>,
    pub dnorm: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearAttentionDeviceParams {
    pub batch: usize,
    pub seq_len: usize,
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
    pub eps: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearAttentionDeviceForwardArgs<'a> {
    pub params: LinearAttentionDeviceParams,
    pub qkv: &'a DeviceHandle,
    pub z: &'a DeviceHandle,
    pub b_proj: &'a DeviceHandle,
    pub a_proj: &'a DeviceHandle,
    pub conv1d_weight: &'a DeviceHandle,
    pub dt_bias: &'a DeviceHandle,
    pub a_log: &'a DeviceHandle,
    pub norm_weight: &'a DeviceHandle,
    // OPD frozen-prompt carry (None = default zero-seed, byte-identical path).
    // initial_state: [batch, num_value_heads, key_dim, value_dim] (final_state layout).
    // initial_conv_window: [batch, conv_kernel-1, qkv_dim].
    pub initial_state: Option<&'a DeviceHandle>,
    pub initial_conv_window: Option<&'a DeviceHandle>,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearAttentionDeviceBoundaryArgs<'a> {
    pub params: LinearAttentionDeviceParams,
    pub qkv: &'a DeviceHandle,
    pub b_proj: &'a DeviceHandle,
    pub a_proj: &'a DeviceHandle,
    pub conv1d_weight: &'a DeviceHandle,
    pub dt_bias: &'a DeviceHandle,
    pub a_log: &'a DeviceHandle,
    pub initial_state: Option<&'a DeviceHandle>,
    pub initial_conv_window: Option<&'a DeviceHandle>,
}

#[derive(Debug, Clone)]
pub struct LinearAttentionDeviceForwardResult {
    pub output: DeviceHandle,
    pub preact: DeviceHandle,
    pub qkv_conv: DeviceHandle,
    pub q: DeviceHandle,
    pub k: DeviceHandle,
    pub v: DeviceHandle,
    pub g: DeviceHandle,
    pub g_cumsum: DeviceHandle,
    pub beta: DeviceHandle,
    pub a_inv: DeviceHandle,
    pub chunk_state: DeviceHandle,
    pub raw_output: DeviceHandle,
    /// The forward took the FlashQLA chunkwise route, so the backward must too.
    /// Recorded on the tape: the runtime flag can flip between calls.
    pub flashqla: bool,
    /// Recurrent state after the last row, `[num_value_heads, key_dim, value_dim]`
    /// f32 — the carry a sequence-parallel successor seeds from.
    pub final_state: DeviceHandle,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearAttentionDeviceBackwardArgs<'a> {
    pub params: LinearAttentionDeviceParams,
    pub upstream: &'a DeviceHandle,
    pub qkv: &'a DeviceHandle,
    pub z: &'a DeviceHandle,
    pub b_proj: &'a DeviceHandle,
    pub a_proj: &'a DeviceHandle,
    pub conv1d_weight: &'a DeviceHandle,
    pub dt_bias: &'a DeviceHandle,
    pub a_log: &'a DeviceHandle,
    pub norm_weight: &'a DeviceHandle,
    pub preact: &'a DeviceHandle,
    pub qkv_conv: &'a DeviceHandle,
    /// None on the FlashQLA route, which re-derives both from `qkv_conv`.
    pub g: Option<&'a DeviceHandle>,
    pub beta: Option<&'a DeviceHandle>,
    /// FlashQLA route: only chunk 0 (= the state carry). Otherwise every chunk.
    pub chunk_state: &'a DeviceHandle,
    /// Some = the FlashQLA route ran; the GDN output the rms-gated backward
    /// differentiates.
    pub raw_output: Option<&'a DeviceHandle>,
    // OPD conv carry (None = default). Feeds the conv1d backward boundary taps'
    // grad_weight; the recurrent state carry lives in chunk_state[0], not here.
    pub initial_conv_window: Option<&'a DeviceHandle>,
    /// Gradient arriving at `final_state` from a successor that seeded from it
    /// (sequence-parallel carry). None = zero.
    pub d_final_state: Option<&'a DeviceHandle>,
}

#[derive(Debug, Clone)]
pub struct LinearAttentionDeviceBackwardResult {
    pub dqkv: DeviceHandle,
    pub dz: DeviceHandle,
    pub db: DeviceHandle,
    pub da: DeviceHandle,
    pub dconv: DeviceHandle,
    pub ddt: DeviceHandle,
    pub da_log: DeviceHandle,
    pub dnorm: DeviceHandle,
    /// d/d(initial_state); Some on the FlashQLA route.
    pub d_initial_state: Option<DeviceHandle>,
    /// d/d(initial_conv_window) `[conv_kernel-1, qkv_dim]` f32; Some when a
    /// window was carried in.
    pub d_initial_conv_window: Option<DeviceHandle>,
}

/// Communicator group: `Seq` = CP subgroup, `Expert` = EP group (both == `World`
/// off a composed mesh).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommAxis {
    World,
    Seq,
    Expert,
}

pub trait Backend: std::fmt::Debug + Send + Sync {
    fn device(&self) -> Device;

    /// Activation storage dtype for tape-saved tensors; CPU stays f32 (parity oracle).
    fn set_tape_dtype(&self, _dtype: crate::TapeDtype) {}

    fn tape_dtype(&self) -> crate::TapeDtype {
        crate::TapeDtype::F32
    }

    /// Drain all in-flight GPU work on this backend's device context
    /// (`cuCtxSynchronize` for CUDA). Default no-op for host/CPU backends.
    ///
    /// Used by the OPD engine time-share (`crates/train/src/opd.rs`) to fence
    /// the *train* backend before an idle infer engine reloads/offloads its
    /// weights. The three co-resident CUDA contexts (infer-student,
    /// infer-teacher, train autograd) share one device async memory pool with
    /// cudarc `disable_event_tracking()` (no automatic cross-stream waits), so
    /// the infer engine's pool allocations during a weight reload can collide
    /// with the train backward's still-outstanding pool ops — handing out the
    /// same physical block twice and dropping reloaded side buffers (observed
    /// as the W4A8 "missing Marlin-packed side buffer" on the teacher reload).
    /// A full train-side device sync orders the train work ahead of the reload.
    fn device_synchronize(&self) -> Result<()> {
        Ok(())
    }

    /// Drain only THIS backend's own default stream (`cuStreamSynchronize` for
    /// CUDA), leaving any co-resident foreign context's streams untouched.
    /// Default no-op for host/CPU backends.
    ///
    /// Unlike [`Backend::device_synchronize`] (`cuCtxSynchronize`, which drains
    /// the entire device primary context), this is the correct fence when a
    /// co-resident inference engine SHARES the device primary context but runs
    /// its streams with cudarc event-tracking disabled and idle-parked between
    /// scheduler steps: a context-wide sync there blocks forever draining the
    /// engine's never-host-progressed streams (the `--share-frozen-base` student
    /// load deadlock). Used for the share-frozen-base cross-stream handoff fence,
    /// which only needs the train backend's own uploads drained — the borrowed
    /// engine base weights were already written by the engine's own load+warmup.
    fn stream_synchronize(&self) -> Result<()> {
        Ok(())
    }

    /// Device VRAM `(free_bytes, total_bytes)` for this backend's context, or
    /// `None` for host/CPU backends with no device memory. Lets OPD log
    /// per-phase resident bytes without a `&CudaBackend` downcast or shelling
    /// out to `nvidia-smi`.
    /// Whether the backend's linear-attention kernels are built for this
    /// (value_heads, key_heads) geometry. CUDA AOT-instantiates per geometry;
    /// host/Metal paths are generic.
    fn linear_attention_head_geometry_supported(&self, _h: usize, _hg: usize) -> bool {
        true
    }

    fn device_mem_info(&self) -> Option<(usize, usize)> {
        None
    }

    /// Async-mempool `(reserved, used)` bytes; `reserved - used` is the hoard
    /// `device_mem_info` can't split from live tensors. `None` off-device.
    fn mem_pool_stats(&self) -> Option<(u64, u64)> {
        None
    }

    /// Hoard in MiB. `None` (not a false 0) if unavailable or `used > reserved`
    /// (a torn two-attr read).
    fn hoarded_mib(&self) -> Option<u64> {
        let (reserved, used) = self.mem_pool_stats()?;
        (reserved >= used).then(|| (reserved - used) >> 20)
    }

    /// Peak pool used-bytes since the last [`Self::reset_mem_pool_used_high`].
    /// The only way to see a transient the hot loop can't sync on. `None`
    /// off-device.
    fn mem_pool_used_high(&self) -> Option<u64> {
        None
    }

    /// Rebase the used-bytes watermark onto current used, scoping the next read
    /// to one phase instead of the whole process.
    fn reset_mem_pool_used_high(&self) -> Result<()> {
        Ok(())
    }

    fn upload(&self, host: &[f32], _shape: &[usize]) -> Result<DeviceHandle> {
        Ok(DeviceHandle::Cpu(host.to_vec()))
    }

    fn upload_bf16_bits(&self, host: &[u16], shape: &[usize]) -> Result<DeviceHandle> {
        if shape_size(shape) != host.len() {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: shape.to_vec(),
                size: shape_size(shape),
            });
        }
        let f32_host: Vec<f32> = host.iter().map(|&bits| bf16_bits_to_f32(bits)).collect();
        self.upload(&f32_host, shape)
    }

    fn upload_fp8_block_scaled(
        &self,
        weight: &[u8],
        scales: &[f32],
        shape: &[usize],
        block_m: usize,
        block_k: usize,
    ) -> Result<DeviceHandle> {
        let f32_host = dequantize_fp8_block_scaled_host(weight, scales, shape, block_m, block_k)?;
        self.upload(&f32_host, shape)
    }

    /// Upload a frozen NVFP4 weight. Backends without a 4-bit lane dequantize on
    /// the host and keep f32; the CUDA backend overrides to stay at 4 bits.
    fn upload_fp4_e2m1_group(
        &self,
        weight: &[u8],
        scales: &[u8],
        global_scale: f32,
        shape: &[usize],
        group_size: usize,
        scale_cols: usize,
    ) -> Result<DeviceHandle> {
        let f32_host = dequantize_fp4_e2m1_group_host(
            weight,
            scales,
            global_scale,
            shape,
            group_size,
            scale_cols,
        )?;
        self.upload(&f32_host, shape)
    }

    /// Import bf16 bytes from a foreign device buffer as an f32 handle.
    ///
    /// `src_stream` is the raw `CUstream` (as `u64`) that produced the source
    /// buffer and will free it. The CUDA backend enqueues the D2D copy on that
    /// stream and orders this backend's stream after it with an event, so the
    /// source may be freed on `src_stream` as soon as this returns. A null
    /// `src_stream` (0) falls back to the legacy synchronous copy.
    fn import_bf16_device_ptr_as_f32(
        &self,
        src_device_ptr: u64,
        src_stream: u64,
        len: usize,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        let _ = (src_device_ptr, src_stream, len, shape);
        Err(crate::AutogradError::TapeInvariant(
            "backend does not support importing bf16 device pointers",
        ))
    }

    /// Re-store a frozen f32 handle as bf16 (`--tape-precision bf16`); consumers
    /// widen on read. Only sound for frozen leaves. Default passthrough; CUDA overrides.
    fn quantize_frozen_to_bf16(
        &self,
        handle: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        let _ = shape;
        Ok(handle.clone())
    }

    /// Import a frozen FP8 block-scaled base weight as a NON-OWNING device view
    /// over buffers owned by a foreign allocator (the infer-cuda rollout/eval
    /// engine's resident base `DeviceMatrix`), for the `--share-frozen-base`
    /// train-infer weight-sharing path.
    ///
    /// `weight_device_ptr` / `scale_device_ptr` are raw CUDA device pointers
    /// (`CUdeviceptr` as `u64`) into the foreign engine's resident FP8 weight
    /// (`rows*cols` u8 bytes, E4M3) and its block scales (`ceil(rows/block_m) *
    /// ceil(cols/block_k)` f32). The returned handle borrows those bytes
    /// **without copying** — the caller must keep the foreign owner resident for
    /// the handle's lifetime (the OPD Phase-B offload must skip the shared base).
    /// The handle is created with `requires_grad = false` semantics by the
    /// caller (frozen base). The default trait impl errors; CUDA overrides.
    fn import_fp8_block_scaled_device_ptr(
        &self,
        weight_device_ptr: u64,
        scale_device_ptr: u64,
        shape: &[usize],
        block_m: usize,
        block_k: usize,
    ) -> Result<DeviceHandle> {
        let _ = (weight_device_ptr, scale_device_ptr, shape, block_m, block_k);
        Err(crate::AutogradError::TapeInvariant(
            "backend does not support importing fp8 block-scaled device pointers",
        ))
    }

    /// Import a NON-OWNING view of a serving engine's NVFP4 base that already
    /// carries the Marlin layout. `global_scale` is the repack's folded value
    /// (2^119 bias and scale_factor divisor included), read as an f32.
    fn import_fp4_marlin_device_ptr(
        &self,
        weight_device_ptr: u64,
        scale_tail_device_ptr: u64,
        global_scale: f32,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        let _ = (
            weight_device_ptr,
            scale_tail_device_ptr,
            global_scale,
            shape,
        );
        Err(crate::AutogradError::TapeInvariant(
            "backend does not support importing marlin nvfp4 device pointers",
        ))
    }

    /// Import a NON-OWNING view of a foreign BF16 device buffer (e.g. the
    /// infer-cuda rollout engine's resident BF16 base `DeviceMatrix` after a
    /// LoRA re-merge), for refreshing the train student's frozen base without
    /// copying ~54 GB.
    ///
    /// `device_ptr` is a raw CUDA device pointer (`CUdeviceptr` as `u64`) into
    /// the foreign engine's resident BF16 weight (`rows*cols` u16 elements).
    /// The returned handle borrows those bytes **without copying** — the caller
    /// must keep the foreign owner resident for the handle's lifetime. The
    /// handle is created with `requires_grad = false` semantics by the caller
    /// (frozen base). The default trait impl errors; CUDA overrides.
    fn import_bf16_device_ptr(&self, device_ptr: u64, shape: &[usize]) -> Result<DeviceHandle> {
        let _ = (device_ptr, shape);
        Err(crate::AutogradError::TapeInvariant(
            "backend does not support importing bf16 device pointers",
        ))
    }

    /// Allocate a zero-filled device handle for `shape`.
    ///
    /// CUDA overrides to allocate and memset on device, which avoids
    /// first-step AdamW moment HtoD traffic.
    fn zeros(&self, shape: &[usize]) -> Result<DeviceHandle> {
        let size = shape_size(shape);
        self.upload(&vec![0.0; size], shape)
    }

    fn readback(&self, handle: &DeviceHandle) -> Result<Vec<f32>> {
        match handle {
            DeviceHandle::Cpu(data) => Ok(data.clone()),
            #[cfg(feature = "metal")]
            DeviceHandle::Metal(_) => Err(crate::AutogradError::TapeInvariant(
                "device handle readback not implemented for metal on this backend",
            )),
            #[cfg(feature = "cuda")]
            DeviceHandle::Cuda(_) => Err(crate::AutogradError::TapeInvariant(
                "device handle readback not implemented for cuda on this backend",
            )),
            #[cfg(feature = "cuda")]
            DeviceHandle::CudaBf16(_) => Err(crate::AutogradError::TapeInvariant(
                "device handle readback not implemented for cuda bf16 on this backend",
            )),
            #[cfg(feature = "cuda")]
            DeviceHandle::CudaFp8BlockScaled(_) => Err(crate::AutogradError::TapeInvariant(
                "device handle readback not implemented for cuda fp8 block-scaled on this backend",
            )),
            #[cfg(feature = "cuda")]
            DeviceHandle::CudaFp4E2M1Group(_) => Err(crate::AutogradError::TapeInvariant(
                "device handle readback not implemented for cuda nvfp4 on this backend",
            )),
        }
    }

    fn readback_into(&self, handle: &DeviceHandle, dst: &mut [f32]) -> Result<()> {
        let src = self.readback(handle)?;
        if src.len() != dst.len() {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: src.len(),
                shape: vec![dst.len()],
                size: dst.len(),
            });
        }
        dst.copy_from_slice(&src);
        Ok(())
    }

    fn eval(&self, _handles: &[&DeviceHandle]) -> Result<()> {
        Ok(())
    }

    /// Park a checkpoint activation in a pinned host slot: one async DtoH on the
    /// backend's stream, no host wait; the slot owns the bytes (write-combined, so
    /// no `Vec<f32>` staging leg). `Ok(None)` = no pool / budget spent → pageable path.
    fn checkpoint_pin_offload(&self, handle: &DeviceHandle, len: usize) -> Result<Option<u32>> {
        let _ = (handle, len);
        Ok(None)
    }

    /// Async HtoD out of `slot` into a fresh handle for `shape`, releasing the slot.
    fn checkpoint_pin_reload(&self, slot: u32, shape: &[usize]) -> Result<DeviceHandle> {
        let _ = (slot, shape);
        Err(crate::AutogradError::TapeInvariant(
            "backend has no pinned checkpoint pool to reload from",
        ))
    }

    /// Blocking read of `slot` into `dst`, releasing the slot.
    fn checkpoint_pin_readback(&self, slot: u32, dst: &mut [f32]) -> Result<()> {
        let _ = (slot, dst);
        Err(crate::AutogradError::TapeInvariant(
            "backend has no pinned checkpoint pool to read back from",
        ))
    }

    /// Return `slot` to the pool unread — its tensor was freed before any reload.
    fn checkpoint_pin_release(&self, slot: u32) {
        let _ = slot;
    }

    /// Optional backend override for Qwen3.5 linear-attention's reverse
    /// state-history scan. The default returns `None` so CPU remains the
    /// reference implementation and non-CUDA backends keep the existing path.
    fn linear_attention_scan_backward(
        &self,
        _args: LinearAttentionScanBackwardArgs<'_>,
    ) -> Result<Option<LinearAttentionScanBackwardGrads>> {
        Ok(None)
    }

    /// Optional device-resident Qwen3.5 gated-delta linear-attention forward.
    /// CPU remains the reference; CUDA overrides exact production shapes.
    fn linear_attention_forward_device(
        &self,
        _args: LinearAttentionDeviceForwardArgs<'_>,
    ) -> Result<Option<LinearAttentionDeviceForwardResult>> {
        Ok(None)
    }

    fn linear_attention_boundary_device(
        &self,
        _args: LinearAttentionDeviceBoundaryArgs<'_>,
    ) -> Result<Option<DeviceHandle>> {
        Ok(None)
    }

    fn linear_attention_backward_device(
        &self,
        _args: LinearAttentionDeviceBackwardArgs<'_>,
    ) -> Result<Option<LinearAttentionDeviceBackwardResult>> {
        Ok(None)
    }

    fn causal_sdpa_recompute_backward_device(
        &self,
        args: CausalSdpaDeviceBackwardArgs<'_>,
    ) -> Result<CausalSdpaDeviceGradTriplet> {
        let q = self.readback(args.q)?;
        let k = self.readback(args.k)?;
        let v = self.readback(args.v)?;
        let upstream = self.readback(args.upstream)?;
        let (grad_q, grad_k, grad_v) = cpu_causal_sdpa_recompute_backward(
            &q,
            &k,
            &v,
            &upstream,
            args.shape,
            args.need_grad_q,
            args.need_grad_k,
            args.need_grad_v,
        )?;
        Ok((
            grad_q
                .as_ref()
                .map(|grad| self.upload(grad, args.shape))
                .transpose()?,
            grad_k
                .as_ref()
                .map(|grad| self.upload(grad, args.shape))
                .transpose()?,
            grad_v
                .as_ref()
                .map(|grad| self.upload(grad, args.shape))
                .transpose()?,
        ))
    }

    fn trim_memory_pool(&self) -> Result<bool> {
        Ok(false)
    }

    /// One context-parallel ring block: fuse this q-tile-set × one KV block into
    /// the running flash-2 `(acc_m, acc_l, acc_o)` accumulator and return the
    /// updated accumulators. `q`/`k_blk`/`v_blk` are f32 handles (converted to
    /// bf16 device-side); `acc_*` are f32. Device-only fast path — CPU/world==1
    /// route through the host `ring_forward_tile` in the ops layer, so this
    /// default never runs there; a non-CUDA caller reaching it is a bug.
    /// `q_pos_host`/`k_pos_host` mirror the device position handles — the FA3
    /// pair decomposition needs them host-side; the scalar kernel ignores them.
    #[allow(clippy::too_many_arguments)]
    fn ring_block_fwd_merge(
        &self,
        _q: &DeviceHandle,
        _k_blk: &DeviceHandle,
        _v_blk: &DeviceHandle,
        _acc_m: &DeviceHandle,
        _acc_l: &DeviceHandle,
        _acc_o: &DeviceHandle,
        _q_pos: &DeviceHandle,
        _k_pos: &DeviceHandle,
        _q_pos_host: &[usize],
        _k_pos_host: &[usize],
        _dims: RingBlockDims,
    ) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
        Err(crate::AutogradError::TapeInvariant(
            "ring_block_fwd_merge is a CUDA-only device path",
        ))
    }

    /// Normalize the ring accumulator after all blocks: `out = O / L` (f32
    /// handle), `lse = M + ln(L)` (f32, one per row). `total_rows = num_q_tiles
    /// * q_rows`.
    fn ring_block_finalize(
        &self,
        _acc_m: &DeviceHandle,
        _acc_l: &DeviceHandle,
        _acc_o: &DeviceHandle,
        _total_rows: usize,
        _head_dim: usize,
    ) -> Result<(DeviceHandle, DeviceHandle)> {
        Err(crate::AutogradError::TapeInvariant(
            "ring_block_finalize is a CUDA-only device path",
        ))
    }

    /// Per-block ring backward (flash-2 adjoint) from the saved `out`/`lse`.
    /// Accumulates into `grad_q` (in/out, returned) and returns this block's
    /// `grad_k_blk`/`grad_v_blk`. `out` is the finalize f32 output; `d_out` is
    /// the upstream grad (f32). Device-only, like `ring_block_fwd_merge`.
    #[allow(clippy::too_many_arguments)]
    fn ring_block_bwd(
        &self,
        _q: &DeviceHandle,
        _k_blk: &DeviceHandle,
        _v_blk: &DeviceHandle,
        _out: &DeviceHandle,
        _lse: &DeviceHandle,
        _d_out: &DeviceHandle,
        _grad_q: &DeviceHandle,
        _q_pos: &DeviceHandle,
        _k_pos: &DeviceHandle,
        _q_pos_host: &[usize],
        _k_pos_host: &[usize],
        _dims: RingBlockDims,
    ) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
        Err(crate::AutogradError::TapeInvariant(
            "ring_block_bwd is a CUDA-only device path",
        ))
    }

    /// Whether `Tape::backward` should `flush_to_host_batch` every
    /// device-resident tape output **before** walking backward. Metal
    /// returns `true` because each `mlx_eval` round-trip dominates at
    /// small shapes and batching N FFI guards into 1 is a real win.
    /// CUDA returns `false` (default) — the batch readback there is the
    /// 1 GB DtoH that per-op lazy readback avoids, and is strictly cheaper because
    /// device-resident downstream backward ops never need the host
    /// snapshot in the first place.
    fn prefers_pre_backward_flush(&self) -> bool {
        false
    }

    /// Compute `C = A @ B` for rank-2 or rank-3 (batched) row-major tensors.
    /// Returns a device handle for the output plus its logical shape.
    fn matmul(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
    ) -> Result<(DeviceHandle, Vec<usize>)>;

    /// Compute `C = A @ B` for rank-2 or rank-3 (batched) row-major tensors.
    /// Returns `(data, output_shape)`. Backends that cannot accelerate a
    /// given shape should fall back to `cpu_matmul_forward`.
    fn matmul_forward(
        &self,
        a: &[f32],
        a_shape: &[usize],
        b: &[f32],
        b_shape: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>)>;

    /// Compute `C = A @ B^T` for rank-2 row-major tensors where
    /// `A:[M,K]`, `B:[N,K]`, and `C:[M,N]`.
    ///
    /// Backends can override to avoid materialising `B^T`.
    fn matmul_bt(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        let a_host = self.readback(a)?;
        let b_host = self.readback(b)?;
        let (out, out_shape) = cpu_matmul_bt_forward(&a_host, a_shape, &b_host, b_shape)?;
        Ok((self.upload(&out, &out_shape)?, out_shape))
    }

    /// Compute the gradients for `C = A @ B` given upstream gradient `dC`.
    /// `need_grad_a`/`need_grad_b` let the caller skip one side; each returned
    /// vector is empty (`vec![]`) if the corresponding `need_grad_*` is false.
    ///
    /// Shapes:
    /// - rank-2: `A:[M,K]`, `B:[K,N]`, `dC:[M,N]`.
    /// - rank-3 (batched): `A:[B,M,K]`, `B:[B,K,N]`, `dC:[B,M,N]`.
    ///
    /// Semantics: `grad_a = dC @ B^T` and `grad_b = A^T @ dC`.
    fn matmul_backward(
        &self,
        a: &[f32],
        a_shape: &[usize],
        b: &[f32],
        b_shape: &[usize],
        grad_out: &[f32],
        grad_out_shape: &[usize],
        need_grad_a: bool,
        need_grad_b: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        cpu_matmul_backward(
            a,
            a_shape,
            b,
            b_shape,
            grad_out,
            grad_out_shape,
            need_grad_a,
            need_grad_b,
        )
    }

    /// Foundation for the device-resident gradient tape.
    ///
    /// Computes `grad_a = grad_out @ B^T` and `grad_b = A^T @ grad_out` and
    /// returns each as an *unevaluated* `DeviceHandle` so the caller can
    /// batch a single terminal `backend.eval(...)` per training step (mirrors
    /// the batched-eval contract used by `adamw_step` /
    /// `log_softmax_last_axis_backward`). `need_grad_a` / `need_grad_b`
    /// short-circuit to `None` so the unused SGEMM is never launched.
    ///
    /// CUDA overrides to keep both SGEMMs on-device with no host roundtrip.
    ///
    /// The existing host-buffer `matmul_backward` stays in place — both
    /// methods coexist while the dispatch wiring lands in a follow-up
    /// subagent.
    fn matmul_backward_device(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
        grad_out: &DeviceHandle,
        grad_out_shape: &[usize],
        need_grad_a: bool,
        need_grad_b: bool,
    ) -> Result<(Option<DeviceHandle>, Option<DeviceHandle>)> {
        if !need_grad_a && !need_grad_b {
            return Ok((None, None));
        }
        let a_host = self.readback(a)?;
        let b_host = self.readback(b)?;
        let grad_host = self.readback(grad_out)?;
        let (grad_a, grad_b) = cpu_matmul_backward(
            &a_host,
            a_shape,
            &b_host,
            b_shape,
            &grad_host,
            grad_out_shape,
            need_grad_a,
            need_grad_b,
        )?;
        let grad_a_handle = if need_grad_a {
            Some(self.upload(&grad_a, a_shape)?)
        } else {
            None
        };
        let grad_b_handle = if need_grad_b {
            Some(self.upload(&grad_b, b_shape)?)
        } else {
            None
        };
        Ok((grad_a_handle, grad_b_handle))
    }

    /// Backward for `C = A @ B^T`.
    fn matmul_bt_backward_device(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
        grad_out: &DeviceHandle,
        grad_out_shape: &[usize],
        need_grad_a: bool,
        need_grad_b: bool,
    ) -> Result<(Option<DeviceHandle>, Option<DeviceHandle>)> {
        if !need_grad_a && !need_grad_b {
            return Ok((None, None));
        }
        let a_host = self.readback(a)?;
        let b_host = self.readback(b)?;
        let grad_host = self.readback(grad_out)?;
        let (grad_a, grad_b) = cpu_matmul_bt_backward(
            &a_host,
            a_shape,
            &b_host,
            b_shape,
            &grad_host,
            grad_out_shape,
            need_grad_a,
            need_grad_b,
        )?;
        let grad_a_handle = if need_grad_a {
            Some(self.upload(&grad_a, a_shape)?)
        } else {
            None
        };
        let grad_b_handle = if need_grad_b {
            Some(self.upload(&grad_b, b_shape)?)
        } else {
            None
        };
        Ok((grad_a_handle, grad_b_handle))
    }

    /// The narrow `grad_a = grad_out @ B` case used by frozen base weights.
    /// It avoids requiring the original `A` handle when no `grad_b` is
    /// needed.
    fn matmul_bt_input_grad_device(
        &self,
        b: &DeviceHandle,
        b_shape: &[usize],
        grad_out: &DeviceHandle,
        grad_out_shape: &[usize],
        input_shape: &[usize],
    ) -> Result<DeviceHandle> {
        let expected_out = matmul_bt_output_shape(input_shape, b_shape)?;
        if grad_out_shape != expected_out.as_slice() {
            return Err(AutogradError::ShapeMismatch {
                expected: expected_out,
                got: grad_out_shape.to_vec(),
            });
        }
        let b_host = self.readback(b)?;
        let grad_host = self.readback(grad_out)?;
        let (grad_a, grad_a_shape) =
            cpu_matmul_forward(&grad_host, grad_out_shape, &b_host, b_shape)?;
        if grad_a_shape != input_shape {
            return Err(AutogradError::ShapeMismatch {
                expected: input_shape.to_vec(),
                got: grad_a_shape,
            });
        }
        self.upload(&grad_a, input_shape)
    }

    /// Elementwise `C = A + B` over identically-shaped contiguous tensors.
    /// Lazy on backends that support it (e.g. Metal defers to `mlx_eval`).
    fn add(&self, a: &DeviceHandle, b: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle>;

    /// Return `dest + src` without changing either input.
    fn add_into_device(
        &self,
        dest: &DeviceHandle,
        src: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        let dest_host = self.readback(dest)?;
        let src_host = self.readback(src)?;
        let size = shape_size(shape);
        if dest_host.len() != size || src_host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: dest_host.len().min(src_host.len()),
                shape: shape.to_vec(),
                size,
            });
        }
        let out: Vec<f32> = dest_host
            .iter()
            .zip(src_host.iter())
            .map(|(d, s)| d + s)
            .collect();
        self.upload(&out, shape)
    }

    /// Accumulate `src` into an exclusively owned persistent gradient.
    fn accumulate_into_device(
        &self,
        dest: &DeviceHandle,
        src: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        self.add_into_device(dest, src, shape)
    }

    /// All-reduce sum over the `axis` communicator. Single-rank and CPU
    /// semantics are identity.
    ///
    /// The operation is functional: it returns a fresh handle and never
    /// mutates `x`, so tape consumers can keep sharing the input handle.
    fn all_reduce_sum_device(
        &self,
        x: &DeviceHandle,
        shape: &[usize],
        axis: CommAxis,
    ) -> Result<DeviceHandle> {
        let _ = axis;
        let host = self.readback(x)?;
        let size = shape_size(shape);
        if host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: shape.to_vec(),
                size,
            });
        }
        self.upload(&host, shape)
    }

    /// All-gather local sequence shards along axis 1 of `[1, S/N, H]` into the
    /// full `[1, S, H]`, concatenated in rank order (context-parallel forward).
    /// `local_shape` is this rank's shard. Single-rank / CPU / no-communicator
    /// semantics are identity (S/N == S), so the default just re-uploads.
    fn all_gather_seq_device(
        &self,
        x: &DeviceHandle,
        local_shape: &[usize],
        axis: CommAxis,
    ) -> Result<DeviceHandle> {
        let _ = axis;
        let host = self.readback(x)?;
        let size = shape_size(local_shape);
        if host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: local_shape.to_vec(),
                size,
            });
        }
        self.upload(&host, local_shape)
    }

    /// Reduce-scatter sum: sum the full-sequence `[1, S, H]` across ranks, keep
    /// this rank's `[1, S/N, H]` row slice — the adjoint of `all_gather_seq`.
    /// `local_shape` is this rank's output shard. Single-rank / CPU / no-communicator
    /// semantics are identity, so the default re-uploads the (already local) input.
    fn reduce_scatter_sum_device(
        &self,
        x: &DeviceHandle,
        local_shape: &[usize],
        axis: CommAxis,
    ) -> Result<DeviceHandle> {
        let _ = axis;
        let host = self.readback(x)?;
        let size = shape_size(local_shape);
        if host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: local_shape.to_vec(),
                size,
            });
        }
        self.upload(&host, local_shape)
    }

    /// One context-parallel ring step: send this rank's KV block to the next rank
    /// and return the block received from the previous rank (a ring rotation of
    /// `[1, kv_heads, block, head_dim]` blocks). `block_shape` is one block's shape
    /// (equal on every rank — the launcher pads the sequence to a multiple of the
    /// CP size). Single-rank / CPU / no-communicator semantics are identity: with
    /// one rank the ring degenerates to the local block, so the default returns the
    /// input. CUDA overrides with `nccl.send`/`recv` inside a `group_start/end`.
    fn ring_send_recv_kv(
        &self,
        block: &DeviceHandle,
        block_shape: &[usize],
    ) -> Result<DeviceHandle> {
        let host = self.readback(block)?;
        let size = shape_size(block_shape);
        if host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: block_shape.to_vec(),
                size,
            });
        }
        self.upload(&host, block_shape)
    }

    /// Point-to-point send of `len` f32 to `peer` on the CP communicator
    /// (sequence-parallel state carry). Caller pairs it with `cp_recv_device`.
    fn cp_send_device(&self, handle: &DeviceHandle, len: usize, peer: usize) -> Result<()> {
        let _ = (handle, len, peer);
        Err(crate::AutogradError::TapeInvariant(
            "cp_send_device: no CP communicator on this backend",
        ))
    }

    /// Point-to-point receive of `len` f32 from `peer` on the CP communicator.
    fn cp_recv_device(&self, len: usize, peer: usize) -> Result<DeviceHandle> {
        let _ = (len, peer);
        Err(crate::AutogradError::TapeInvariant(
            "cp_recv_device: no CP communicator on this backend",
        ))
    }

    /// All-to-all: split `scatter_axis` across ranks, concatenate each rank's
    /// slice along `gather_axis`. Returns `(handle, out_shape)` — the shape
    /// changes (`[seq/N,b,hidden]` → `[seq,b,hidden/N]`), so the caller can't
    /// derive it. Single-rank / CPU / no-communicator semantics are identity
    /// (out_shape == in_shape); the default returns the input unchanged.
    fn all_to_all_device(
        &self,
        x: &DeviceHandle,
        in_shape: &[usize],
        scatter_axis: usize,
        gather_axis: usize,
        axis: CommAxis,
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        let _ = (scatter_axis, gather_axis, axis);
        let host = self.readback(x)?;
        let size = shape_size(in_shape);
        if host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: in_shape.to_vec(),
                size,
            });
        }
        Ok((self.upload(&host, in_shape)?, in_shape.to_vec()))
    }

    /// Sum of squares for a device handle, returned on host as `f64`.
    /// CUDA overrides with a partial-reduction kernel so gradient clipping
    /// can stay device-resident.
    fn sum_squares(&self, x: &DeviceHandle, shape: &[usize]) -> Result<f64> {
        let host = self.readback(x)?;
        let size = shape_size(shape);
        if host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: shape.to_vec(),
                size,
            });
        }
        Ok(host
            .iter()
            .map(|&value| {
                let value = f64::from(value);
                value * value
            })
            .sum())
    }

    /// Global-norm gradient clip across many tensors.
    ///
    /// Default returns `None` so higher-level train code can fall back to the
    /// portable per-tensor path. CUDA overrides with a batched pointer-array
    /// reduction plus batched scale kernel.
    fn clip_grad_norm_device(
        &self,
        grads: &[(DeviceHandle, Vec<usize>)],
        max_norm: f32,
    ) -> Result<Option<DeviceGradClipResult>> {
        let _ = (grads, max_norm);
        Ok(None)
    }

    /// Reduce-sum **all** elements of `x` into a rank-0 scalar device handle.
    /// `shape` describes the input layout (`product(shape)` elements; an
    /// empty shape means a 1-element scalar).
    ///
    /// Lazy on backends that support it: Metal composes this into the MLX
    /// graph (`reshape -> sum_axis(0)`) and defers `mlx_eval` to whatever
    /// terminal op forces a host readback. CPU/CUDA remain eager and return
    /// a fully-realized handle.
    fn sum_all(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle>;

    /// Row-wise softmax over the last dim. `shape` describes a contiguous
    /// tensor of rank ≥ 1; softmax is applied along the final axis.
    fn softmax_forward_last_axis(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
        cpu_softmax_forward_last_axis(x, shape)
    }

    /// Row-wise log-softmax over the last dim. Numerically stable
    /// (subtract max, log-sum-exp) — mirrors `ops::softmax::log_softmax`.
    fn log_softmax_forward_last_axis(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
        cpu_log_softmax_forward_last_axis(x, shape)
    }

    /// Lazy on backends that can compose softmax into their graph (Metal:
    /// `mlx_softmax_axis`).
    fn softmax_last_axis(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.softmax_forward_last_axis(&host, shape)?;
        self.upload(&out, shape)
    }

    /// Lazy on backends that can compose into their graph (Metal uses
    /// `mlx_logsumexp_axis` + `mlx_subtract`).
    fn log_softmax_last_axis(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.log_softmax_forward_last_axis(&host, shape)?;
        self.upload(&out, shape)
    }

    /// Computes `grad_input = y * (upstream - sum(upstream * y, axis=-1, keepdim=true))`
    /// row-wise over the last axis, where `y` is the saved forward softmax
    /// output.
    fn softmax_last_axis_backward(
        &self,
        upstream: &DeviceHandle,
        softmax_output: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        let upstream_host = self.readback(upstream)?;
        let output_host = self.readback(softmax_output)?;
        let grad = cpu_softmax_backward(&upstream_host, &output_host, shape)?;
        self.upload(&grad, shape)
    }

    /// Computes `grad_input = upstream - exp(log_softmax_output) * sum(upstream, axis=-1, keepdim=true)`
    /// row-wise over the last axis.
    ///
    /// `log_softmax_output` is the saved forward output (NOT the input —
    /// `softmax(x) = exp(log_softmax(x))` and the backward identity uses the
    /// softmax probability, which is just `exp(saved_output)`). `upstream`
    /// has the same shape as `log_softmax_output`.
    ///
    /// CUDA overrides this with a single per-row NVRTC kernel that consumes
    /// the saved forward output without a host roundtrip — kills the
    /// `[B, S, V]` × 4 B ≈ 1 GB DtoH copy that nsys identified as the
    /// single largest readback per training step.
    fn log_softmax_last_axis_backward(
        &self,
        upstream: &DeviceHandle,
        log_softmax_output: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        let upstream_host = self.readback(upstream)?;
        let output_host = self.readback(log_softmax_output)?;
        let grad = cpu_log_softmax_backward(&upstream_host, &output_host, shape)?;
        self.upload(&grad, shape)
    }

    /// Lazy on backends that can compose `x * sigmoid(x)` into their graph
    /// (Metal: `mlx_multiply` + `mlx_sigmoid`).
    fn silu(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.silu_forward(&host)?;
        self.upload(&out, shape)
    }

    /// Lazy on backends with a native `exp` graph node (Metal: `mlx_exp`).
    fn exp(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.exp_forward(&host)?;
        self.upload(&out, shape)
    }

    /// Elementwise `out = 1 / (1 + exp(-a))`.
    fn sigmoid_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        cpu_sigmoid_forward(a)
    }

    /// Lazy on backends with a native `sigmoid` graph node (Metal: `mlx_sigmoid`).
    fn sigmoid(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.sigmoid_forward(&host)?;
        self.upload(&out, shape)
    }

    /// Elementwise `out = |a|`.
    fn abs_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        cpu_abs_forward(a)
    }

    /// CUDA overrides with a 1D kernel.
    fn abs(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.abs_forward(&host)?;
        self.upload(&out, shape)
    }

    /// Elementwise `out = a * b` over identically-sized contiguous tensors.
    fn mul_forward(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        cpu_mul_forward(a, b)
    }

    /// Lazy on backends that can compose `a * b` into their graph (Metal:
    /// `mlx_multiply`). Shapes must match on both sides (elementwise, not
    /// broadcasted — use `add_broadcast`'s `mul` twin if broadcast
    /// multiplication is ever needed).
    fn mul(&self, a: &DeviceHandle, b: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let a_host = self.readback(a)?;
        let b_host = self.readback(b)?;
        let out = self.mul_forward(&a_host, &b_host)?;
        self.upload(&out, shape)
    }

    /// Elementwise `out = a * s` for scalar `s`.
    fn mul_scalar_forward(&self, a: &[f32], s: f32) -> Result<Vec<f32>> {
        cpu_mul_scalar_forward(a, s)
    }

    /// Lazy on backends that can compose `x * s` into their graph (Metal:
    /// broadcast `mlx_multiply` against a rank-0 scalar `mlx_array`).
    fn mul_scalar(&self, x: &DeviceHandle, s: f32, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.mul_scalar_forward(&host, s)?;
        self.upload(&out, shape)
    }

    /// Computes `grad_x[i] = upstream[i] * k` and returns an unevaluated
    /// handle.
    ///
    /// CUDA overrides with a 1D NVRTC kernel.
    ///
    /// Wires the CE-loss backward chain the surrounding device overrides
    /// already cover: `mul_scalar_backward` was the *first* host op in
    /// `d_loss → mul_scalar_backward → mean_backward → gather_backward →
    /// log_softmax_backward → matmul_backward`, so its host fallback
    /// demoted every downstream `device_path_ok` gate to host. Keeping
    /// this on-device unblocks the whole chain.
    fn mul_scalar_backward_device(
        &self,
        upstream_grad: &DeviceHandle,
        scale: f32,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        let upstream_host = self.readback(upstream_grad)?;
        let grad = self.mul_scalar_forward(&upstream_host, scale)?;
        self.upload(&grad, shape)
    }

    /// The forward reduces `elem_count = product(output_shape)` elements to
    /// a rank-0 scalar; the backward broadcasts
    /// `upstream_grad / elem_count` across `elem_count` slots of the
    /// returned `d_input` handle.
    ///
    /// `upstream_grad` must be a rank-0 scalar (shape `[]` or `[1]`).
    /// `output_shape` is the shape of the input to the original `mean`
    /// op (i.e. the shape of the returned `d_input`).
    ///
    /// CUDA overrides with a 1D NVRTC kernel that fetches the upstream
    /// scalar from device memory (free L1 broadcast) and writes one slot
    /// per thread.
    ///
    /// Pairs with `mul_scalar_backward_device` to keep the CE-loss
    /// backward chain device-resident.
    fn mean_backward_device(
        &self,
        upstream_grad: &DeviceHandle,
        output_shape: &[usize],
        elem_count: usize,
    ) -> Result<DeviceHandle> {
        let upstream_host = self.readback(upstream_grad)?;
        if upstream_host.len() != 1 {
            return Err(crate::AutogradError::ShapeMismatch {
                expected: Vec::new(),
                got: vec![upstream_host.len()],
            });
        }
        let inv = if elem_count == 0 {
            0.0
        } else {
            1.0 / elem_count as f32
        };
        let value = upstream_host[0] * inv;
        let grad = vec![value; elem_count];
        self.upload(&grad, output_shape)
    }

    /// Broadcast the rank-0 upstream scalar across the original input
    /// shape.
    fn sum_backward_device(
        &self,
        upstream_grad: &DeviceHandle,
        output_shape: &[usize],
    ) -> Result<DeviceHandle> {
        let upstream_host = self.readback(upstream_grad)?;
        if upstream_host.len() != 1 {
            return Err(crate::AutogradError::ShapeMismatch {
                expected: Vec::new(),
                got: vec![upstream_host.len()],
            });
        }
        let elem_count = shape_size(output_shape);
        let grad = vec![upstream_host[0]; elem_count];
        self.upload(&grad, output_shape)
    }

    /// Right-aligned broadcast-add `out[i..] = a[i..] + b[broadcast_offset(i)]`.
    ///
    /// `b_shape.len() <= a_shape.len()`. Each `b`-axis of size 1 broadcasts
    /// across the corresponding `a`-axis; otherwise the size must match.
    /// Output shape equals `a_shape`.
    fn add_broadcast_forward(
        &self,
        a: &[f32],
        a_shape: &[usize],
        b: &[f32],
        b_shape: &[usize],
    ) -> Result<Vec<f32>> {
        cpu_add_broadcast_forward(a, a_shape, b, b_shape)
    }

    /// Lazy on backends whose native add already broadcasts (Metal:
    /// `mlx_add` — NumPy-style right-aligned broadcasting, no explicit
    /// `broadcast_to` needed). Output shape equals `a_shape`.
    fn add_broadcast(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
    ) -> Result<DeviceHandle> {
        let a_host = self.readback(a)?;
        let b_host = self.readback(b)?;
        let out = self.add_broadcast_forward(&a_host, a_shape, &b_host, b_shape)?;
        self.upload(&out, a_shape)
    }

    /// Broadcast-copy `src` up to `target_shape` (host): a pure right-aligned
    /// expand, `out[i] = src[broadcast_offset(i)]` — no zero carrier.
    fn broadcast_expand_forward(
        &self,
        src: &[f32],
        src_shape: &[usize],
        target_shape: &[usize],
    ) -> Result<Vec<f32>> {
        validate_broadcast(target_shape, src_shape)?;
        let total = shape_size(target_shape);
        let mut out = vec![0.0_f32; total];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = src[broadcast_offset(i, target_shape, src_shape)];
        }
        Ok(out)
    }

    /// CUDA/Metal override with an in-place device expand (no zero carrier,
    /// no round-trip).
    fn broadcast_expand(
        &self,
        src: &DeviceHandle,
        src_shape: &[usize],
        target_shape: &[usize],
    ) -> Result<DeviceHandle> {
        let src_host = self.readback(src)?;
        let out = self.broadcast_expand_forward(&src_host, src_shape, target_shape)?;
        self.upload(&out, target_shape)
    }

    /// Elementwise `out = exp(a)`.
    fn exp_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        cpu_exp_forward(a)
    }

    /// Elementwise `out = -a`.
    fn neg_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        cpu_neg_forward(a)
    }

    /// Elementwise GELU (tanh approximation), matches `ops::activation::gelu`.
    fn gelu_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        cpu_gelu_forward(a)
    }

    /// Elementwise SiLU (Swish) — `out = a * sigmoid(a)`.
    fn silu_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        cpu_silu_forward(a)
    }

    /// Row-wise RMSNorm over the last axis. `weight` has length = last_dim;
    /// `x` is a contiguous tensor of any rank ≥ 1 with last dim matching.
    fn rms_norm_forward(
        &self,
        x: &[f32],
        weight: &[f32],
        shape: &[usize],
        eps: f32,
    ) -> Result<Vec<f32>> {
        cpu_rms_norm_forward(x, weight, shape, eps)
    }

    /// Lazy on backends with a native fused rms-norm op (Metal:
    /// `mlx_fast_rms_norm` over a borrowed `x` handle + per-call `weight`
    /// upload). Backward path recomputes `inv_rms` host-side — see
    /// `ops::norm` for the saved-context encoding.
    fn rms_norm(
        &self,
        x: &DeviceHandle,
        weight: &[f32],
        shape: &[usize],
        eps: f32,
    ) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.rms_norm_forward(&host, weight, shape, eps)?;
        self.upload(&out, shape)
    }

    /// Gather embedding rows by token ids.
    /// `weight` is `[vocab, dim]` row-major; `ids` has length `n_ids`.
    /// Returns a contiguous `[n_ids * dim]` buffer shaped by the caller.
    fn embedding_forward(
        &self,
        weight: &[f32],
        vocab: usize,
        dim: usize,
        ids: &[i32],
    ) -> Result<Vec<f32>> {
        cpu_embedding_forward(weight, vocab, dim, ids)
    }

    /// Lazy on backends that can compose the row-gather into their eval
    /// stream (Metal: upload `ids` as a tiny int32 array, `mlx_take_axis`,
    /// reshape, no eval). Output shape is `[1, ids.len(), dim]`. This
    /// matches the `ops::embedding` convention of treating raw ids as a
    /// single batch row.
    fn embedding(
        &self,
        table: &DeviceHandle,
        table_shape: &[usize],
        ids: &[i32],
    ) -> Result<DeviceHandle> {
        if table_shape.len() != 2 {
            return Err(crate::AutogradError::InvalidRank {
                expected: "2",
                got: table_shape.len(),
            });
        }
        let vocab = table_shape[0];
        let hidden = table_shape[1];
        let host = self.readback(table)?;
        let out = self.embedding_forward(&host, vocab, hidden, ids)?;
        self.upload(&out, &[1, ids.len(), hidden])
    }

    /// Device-token embedding variant for greedy decode loops. `ids` is a
    /// device-resident f32 vector whose values are exact integer token ids.
    fn embedding_from_f32_ids(
        &self,
        table: &DeviceHandle,
        table_shape: &[usize],
        ids: &DeviceHandle,
        n_ids: usize,
    ) -> Result<DeviceHandle> {
        let ids_host = self.readback(ids)?;
        if ids_host.len() != n_ids {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: ids_host.len(),
                shape: vec![n_ids],
                size: n_ids,
            });
        }
        let ids_i32 = ids_host.iter().map(|&id| id as i32).collect::<Vec<_>>();
        self.embedding(table, table_shape, &ids_i32)
    }

    /// Argmax over the last axis. Returns f32 indices shaped as
    /// `[product(shape[..-1])]` so the existing f32-only DeviceHandle can
    /// carry rollout token ids without adding an integer storage variant.
    fn argmax_last_dim(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let vocab = *shape.last().ok_or(crate::AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        })?;
        if vocab == 0 {
            return Err(crate::AutogradError::InvalidRank {
                expected: "non-empty last dim",
                got: 0,
            });
        }
        let rows = shape_size(shape) / vocab;
        let out: Vec<f32> = host
            .chunks(vocab)
            .take(rows)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0.0, |(idx, _)| idx as f32)
            })
            .collect();
        self.upload(&out, &[rows])
    }

    /// Return a new copy of `dest` with `src[0]` written at `index`.
    fn write_scalar_at(
        &self,
        dest: &DeviceHandle,
        src: &DeviceHandle,
        len: usize,
        index: usize,
    ) -> Result<DeviceHandle> {
        if index >= len {
            return Err(crate::AutogradError::IndexOutOfBounds { index, upper: len });
        }
        let mut host = self.readback(dest)?;
        let src_host = self.readback(src)?;
        if host.len() != len || src_host.is_empty() {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: host.len(),
                shape: vec![len],
                size: len,
            });
        }
        host[index] = src_host[0];
        self.upload(&host, &[len])
    }

    /// Lazy GELU (erf form), matching `ops::activation::gelu`'s CPU body:
    /// `0.5 * x * (1 + erf(x / sqrt(2)))`. NOT the tanh-approx variant
    /// exposed by `gelu_forward` — those two formulas differ at the ~1e-3
    /// level, and `gelu_backward` hard-codes the erf-derivative via the
    /// saved input, so forward must stay on the erf form for the
    /// saved-input derivative to be consistent.
    fn gelu(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out: Vec<f32> = host
            .iter()
            .map(|&value| 0.5 * value * (1.0 + libm::erff(value * 0.707_106_77)))
            .collect();
        self.upload(&out, shape)
    }

    /// Reduce-sum over the last axis. Output has length `product(shape[..-1])`
    /// (or 1 if `shape.len() == 1`).
    fn sum_last_axis_forward(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
        cpu_sum_last_axis_forward(x, shape)
    }

    /// Reduce-mean over the last axis.
    fn mean_last_axis_forward(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
        cpu_mean_last_axis_forward(x, shape)
    }

    /// Rotary position embedding (NeoX / `rotate_half` layout, matches Qwen3.5).
    /// `x` is `[batch, heads, seq, head_dim]`; `cos`/`sin` are
    /// `[seq, rotary_dim/2]`, where `rotary_dim <= head_dim`. When
    /// `rotary_dim < head_dim`, only the prefix is rotated and the suffix is
    /// copied through unchanged.
    fn rope_forward(
        &self,
        x: &[f32],
        x_shape: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Vec<f32>> {
        cpu_rope_forward(x, x_shape, cos, sin)
    }

    /// Lazy on backends that can compose the half-split rotation graph into
    /// their eval stream (Metal: `mlx_slice` → `mlx_multiply` →
    /// `mlx_subtract`/`mlx_add` → `mlx_concatenate_axis`, no eval).
    /// `cos`/`sin` stay as host slices — the caches are precomputed per seq
    /// length and seldom benefit from being device-resident, and keeping
    /// them host-side means no merge of device handles is required.
    fn rope(
        &self,
        x: &DeviceHandle,
        x_shape: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.rope_forward(&host, x_shape, cos, sin)?;
        self.upload(&out, x_shape)
    }

    /// Gather along the last axis: `out[prefix] = src[prefix * vocab + ids[prefix]]`.
    /// `src_shape[..-1]` dictates the prefix shape; `ids.len()` must equal the
    /// prefix product. The caller is expected to have bounds-checked the ids.
    fn gather_last_dim_forward(
        &self,
        src: &[f32],
        src_shape: &[usize],
        ids: &[i32],
    ) -> Result<Vec<f32>> {
        cpu_gather_last_dim_forward(src, src_shape, ids)
    }

    /// Lazy on backends that can compose `flatten → take_axis → reshape`
    /// into their eval stream (Metal: `mlx_reshape` to `[prefix*vocab]`,
    /// `mlx_take_axis` with remapped `i * vocab + ids[i]` flat ids,
    /// `mlx_reshape` back to `src_shape[..-1]`). Output shape is
    /// `src_shape[..-1]` (empty for rank-1 input).
    fn gather_last_dim(
        &self,
        src: &DeviceHandle,
        src_shape: &[usize],
        ids: &[i32],
    ) -> Result<DeviceHandle> {
        let host = self.readback(src)?;
        let out = self.gather_last_dim_forward(&host, src_shape, ids)?;
        let out_shape: Vec<usize> = if src_shape.len() <= 1 {
            Vec::new()
        } else {
            src_shape[..src_shape.len() - 1].to_vec()
        };
        self.upload(&out, &out_shape)
    }

    /// Zero-fills a `[prefix_rows, vocab] = src_shape` output and scatters
    /// the per-prefix `upstream` values into the `(row, ids[row])` slots.
    /// Equivalent to the flat
    /// `scatter_add_rows_forward(upstream, prefix_rows, 1, remapped_ids,
    /// prefix_rows * vocab)` path the host backward takes, but the
    /// trait-level signature exposes the natural `[B, S, V]` output shape
    /// so backends can pick block tiling (one block per prefix row of
    /// `vocab` cols) without un-flattening.
    ///
    /// `upstream` has length `product(src_shape[..-1])` (one scalar per
    /// prefix position). `indices.len() == prefix_rows == upstream.len()`.
    /// Negative or out-of-range indices are silently skipped, matching
    /// `cpu_gather_last_dim_backward` / `cpu_scatter_add_rows_forward`.
    ///
    /// CUDA overrides this with a single per-row NVRTC kernel so the
    /// `[B, S, V]` grad stays device-resident — keeps the `1 GB`
    /// scatter-add output off the host-roundtrip path the host
    /// `gather_last_dim_backward` forces.
    fn gather_last_dim_backward(
        &self,
        upstream: &DeviceHandle,
        indices: &[i32],
        src_shape: &[usize],
    ) -> Result<DeviceHandle> {
        let upstream_host = self.readback(upstream)?;
        let grad = cpu_gather_last_dim_backward(&upstream_host, indices, src_shape)?;
        self.upload(&grad, src_shape)
    }

    /// Pure-layout reshape: returns a handle whose view is `new_shape` over
    /// the same logical elements. Numel must match; the caller is expected
    /// to have checked that. Metal overrides to `mlx_reshape` so the whole
    /// graph stays lazy — reshape is a free metadata op on MLX side.
    fn reshape(&self, x: &DeviceHandle, new_shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        self.upload(&host, new_shape)
    }

    /// Swap two axes of `x`. `old_shape` is the pre-swap shape; the caller is
    /// responsible for computing the post-swap shape (just swap the two
    /// entries). `axis1`/`axis2` must be valid axes into `old_shape`. Metal
    /// overrides to `mlx_transpose_axes` with a permutation that is identity
    /// except for the two swapped positions, composing into the lazy graph.
    fn transpose_axes_swap(
        &self,
        x: &DeviceHandle,
        old_shape: &[usize],
        axis1: usize,
        axis2: usize,
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        let host = self.readback(x)?;
        let (data, new_shape) = cpu_transpose_swap(&host, old_shape, axis1, axis2)?;
        let handle = self.upload(&data, &new_shape)?;
        Ok((handle, new_shape))
    }

    /// Contiguous-stride slice of `x` over `old_shape` from per-axis `starts`
    /// (inclusive) to `ends` (exclusive). Returns a new device handle whose
    /// logical shape is `ends - starts` (caller computes). Metal overrides to
    /// `mlx_slice` with strides=1, wrapping the non-contiguous view in
    /// `mlx_contiguous` so readback respects the sliced window (same
    /// rationale as the `transpose_axes_swap` override).
    fn slice(
        &self,
        x: &DeviceHandle,
        old_shape: &[usize],
        starts: &[usize],
        ends: &[usize],
    ) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let (data, new_shape) = cpu_slice(&host, old_shape, starts, ends)?;
        self.upload(&data, &new_shape)
    }

    /// Concatenate two rank-4 `[batch, heads, seq, dim]` tensors along the
    /// sequence axis. CUDA overrides this for OPD rollout KV-cache appends
    /// so cached K/V stay device-resident during greedy decode.
    fn concat_axis2(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        let a_host = self.readback(a)?;
        let b_host = self.readback(b)?;
        let (data, out_shape) = cpu_concat_axis2(&a_host, a_shape, &b_host, b_shape)?;
        let handle = self.upload(&data, &out_shape)?;
        Ok((handle, out_shape))
    }

    /// Concatenate N same-rank tensors along `axis` (shapes equal off `axis`).
    /// CUDA overrides D2D (the CP reorder concats a full-seq tensor per layer
    /// — a host round-trip there costs hours at 256K).
    fn concat(
        &self,
        parts: &[(&DeviceHandle, &[usize])],
        axis: usize,
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        let first = parts[0].1;
        let outer: usize = first[..axis].iter().product();
        let inner: usize = first[axis + 1..].iter().product();
        let axis_total: usize = parts.iter().map(|(_, s)| s[axis]).sum();
        let mut out_shape = first.to_vec();
        out_shape[axis] = axis_total;
        let mut data = vec![0.0f32; outer * axis_total * inner];
        let mut axis_off = 0usize;
        for (handle, shape) in parts {
            let host = self.readback(handle)?;
            let axis_i = shape[axis];
            for o in 0..outer {
                let src_base = o * axis_i * inner;
                let dst_base = (o * axis_total + axis_off) * inner;
                let len = axis_i * inner;
                data[dst_base..dst_base + len].copy_from_slice(&host[src_base..src_base + len]);
            }
            axis_off += axis_i;
        }
        let handle = self.upload(&data, &out_shape)?;
        Ok((handle, out_shape))
    }

    /// Write compact rank-4 `[batch, heads, src_seq, dim]` `src` into the
    /// sequence window of a preallocated `[batch, heads, max_seq, dim]` cache.
    ///
    /// CUDA overrides this with an in-place kernel and returns a clone of the
    /// destination handle. The CPU fallback returns a fresh uploaded handle so
    /// the functional contract remains backend-neutral.
    fn kv_cache_write_axis2(
        &self,
        dst: &DeviceHandle,
        dst_shape: &[usize],
        src: &DeviceHandle,
        src_shape: &[usize],
        seq_offset: usize,
    ) -> Result<DeviceHandle> {
        let mut dst_host = self.readback(dst)?;
        let src_host = self.readback(src)?;
        cpu_kv_cache_write_axis2(&mut dst_host, dst_shape, &src_host, src_shape, seq_offset)?;
        self.upload(&dst_host, dst_shape)
    }

    /// Decode-time GQA causal attention for a one-token query:
    /// `out = softmax(q @ k^T / sqrt(D)) @ v`.
    ///
    /// Shapes:
    /// - `q`: `[batch, query_heads, 1, head_dim]`
    /// - `k`/`v`: `[batch, kv_heads, kv_len, head_dim]`
    ///
    /// This is the narrow OPD rollout fast path. `query_heads` may be a
    /// multiple of `kv_heads`; each query head maps to `kv_head =
    /// query_head / (query_heads / kv_heads)`. The default fallback keeps
    /// non-CUDA backends correct by using the CPU reference.
    fn causal_sdpa_decode_gqa(
        &self,
        q: &DeviceHandle,
        q_shape: &[usize],
        k: &DeviceHandle,
        k_shape: &[usize],
        v: &DeviceHandle,
        v_shape: &[usize],
        q_start: usize,
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        let q_host = self.readback(q)?;
        let k_host = self.readback(k)?;
        let v_host = self.readback(v)?;
        let (data, out_shape) = cpu_causal_sdpa_decode_gqa(
            &q_host, q_shape, &k_host, k_shape, &v_host, v_shape, q_start,
        )?;
        let handle = self.upload(&data, &out_shape)?;
        Ok((handle, out_shape))
    }

    /// Fused causal SDPA prefill forward (flash-style online softmax — no
    /// `[seq, seq]` score transient). `q` is `[1, q_heads, q_len, head_dim]`,
    /// `k`/`v` `[1, kv_heads, kv_len, head_dim]` with `kv_len = q_start +
    /// q_len`; GQA native. `None` => no fused path for this backend/shape and
    /// the caller composes from primitives.
    #[allow(clippy::too_many_arguments)]
    fn causal_sdpa_prefill_device(
        &self,
        q: &DeviceHandle,
        q_shape: &[usize],
        k: &DeviceHandle,
        k_shape: &[usize],
        v: &DeviceHandle,
        v_shape: &[usize],
        q_start: usize,
    ) -> Result<Option<(DeviceHandle, Vec<usize>)>> {
        let _ = (q, q_shape, k, k_shape, v, v_shape, q_start);
        Ok(None)
    }

    /// Decode-time GQA attention over a preallocated KV cache. `k_shape` and
    /// `v_shape` are the full cache shapes `[batch, kv_heads, max_seq, dim]`;
    /// `kv_len` declares how many prefix tokens are valid.
    fn causal_sdpa_decode_gqa_cache(
        &self,
        q: &DeviceHandle,
        q_shape: &[usize],
        k: &DeviceHandle,
        k_shape: &[usize],
        v: &DeviceHandle,
        v_shape: &[usize],
        kv_len: usize,
        q_start: usize,
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        let q_host = self.readback(q)?;
        let k_host = self.readback(k)?;
        let v_host = self.readback(v)?;
        let (data, out_shape) = cpu_causal_sdpa_decode_gqa_cache(
            &q_host, q_shape, &k_host, k_shape, &v_host, v_shape, kv_len, q_start,
        )?;
        let handle = self.upload(&data, &out_shape)?;
        Ok((handle, out_shape))
    }

    /// Decode-only Qwen attention preparation for the rollout fast path.
    ///
    /// Inputs are post-projection tensors with shape `[batch, 1, out_dim]`.
    /// The output `q` has shape `[batch, query_heads, 1, head_dim]`; when
    /// `gated` is true, the returned gate has the same shape and contains the
    /// raw gate half in head-major layout. This fuses the decode-only
    /// split/reshape/transpose/RMSNorm/RoPE chain while preserving the
    /// existing gate order (`sigmoid(gate) * attn_hidden` happens later).
    #[allow(clippy::too_many_arguments)]
    fn qwen_decode_prepare_q(
        &self,
        q_full: &DeviceHandle,
        q_full_shape: &[usize],
        q_norm_weight: &DeviceHandle,
        q_norm_weight_shape: &[usize],
        cos: &DeviceHandle,
        cos_shape: &[usize],
        sin: &DeviceHandle,
        sin_shape: &[usize],
        query_heads: usize,
        head_dim: usize,
        gated: bool,
        eps: f32,
    ) -> Result<(DeviceHandle, Option<DeviceHandle>, Vec<usize>)> {
        let q_full_host = self.readback(q_full)?;
        let weight_host = self.readback(q_norm_weight)?;
        let cos_host = self.readback(cos)?;
        let sin_host = self.readback(sin)?;
        let (q, gate, out_shape) = cpu_qwen_decode_prepare_q(
            &q_full_host,
            q_full_shape,
            &weight_host,
            q_norm_weight_shape,
            &cos_host,
            cos_shape,
            &sin_host,
            sin_shape,
            query_heads,
            head_dim,
            gated,
            eps,
        )?;
        let q_handle = self.upload(&q, &out_shape)?;
        let gate_handle = gate
            .map(|gate| self.upload(&gate, &out_shape))
            .transpose()?;
        Ok((q_handle, gate_handle, out_shape))
    }

    /// Decode-only Qwen K/V preparation for the rollout fast path.
    ///
    /// Inputs are post-projection tensors with shape `[batch, 1,
    /// kv_heads * head_dim]`. The returned K/V tensors have shape
    /// `[batch, kv_heads, 1, head_dim]`; K is RMSNorm + RoPE transformed,
    /// V is only laid out head-major.
    #[allow(clippy::too_many_arguments)]
    fn qwen_decode_prepare_kv(
        &self,
        k_full: &DeviceHandle,
        k_full_shape: &[usize],
        v_full: &DeviceHandle,
        v_full_shape: &[usize],
        k_norm_weight: &DeviceHandle,
        k_norm_weight_shape: &[usize],
        cos: &DeviceHandle,
        cos_shape: &[usize],
        sin: &DeviceHandle,
        sin_shape: &[usize],
        kv_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<(DeviceHandle, DeviceHandle, Vec<usize>)> {
        let k_full_host = self.readback(k_full)?;
        let v_full_host = self.readback(v_full)?;
        let weight_host = self.readback(k_norm_weight)?;
        let cos_host = self.readback(cos)?;
        let sin_host = self.readback(sin)?;
        let (k, v, out_shape) = cpu_qwen_decode_prepare_kv(
            &k_full_host,
            k_full_shape,
            &v_full_host,
            v_full_shape,
            &weight_host,
            k_norm_weight_shape,
            &cos_host,
            cos_shape,
            &sin_host,
            sin_shape,
            kv_heads,
            head_dim,
            eps,
        )?;
        let k_handle = self.upload(&k, &out_shape)?;
        let v_handle = self.upload(&v, &out_shape)?;
        Ok((k_handle, v_handle, out_shape))
    }

    /// Returns a full `old_shape` gradient with upstream values scattered
    /// into the sliced window and zeros elsewhere.
    fn slice_backward_device(
        &self,
        upstream: &DeviceHandle,
        input_shape: &[usize],
        starts: &[usize],
        ends: &[usize],
    ) -> Result<DeviceHandle> {
        let upstream_host = self.readback(upstream)?;
        let expected_shape = validate_slice_shape(input_shape, starts, ends)?;
        let expected_size = shape_size(&expected_shape);
        if upstream_host.len() != expected_size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: upstream_host.len(),
                shape: expected_shape,
                size: expected_size,
            });
        }

        let input_strides = broadcast_strides(input_shape);
        let mut grad = vec![0.0; shape_size(input_shape)];
        for (out_index, &grad_value) in upstream_host.iter().enumerate() {
            let out_coords = linear_to_coords(out_index, &expected_shape);
            let input_index: usize = out_coords
                .iter()
                .enumerate()
                .map(|(axis, &coord)| (coord + starts[axis]) * input_strides[axis])
                .sum();
            grad[input_index] += grad_value;
        }
        self.upload(&grad, input_shape)
    }

    /// Reorder whole seq blocks: destination block `i` takes source block
    /// `perm[i]`. Blocks are contiguous element ranges, so a backend can move
    /// them without materializing one tensor per block.
    fn permute_seq_blocks_device(
        &self,
        x: &DeviceHandle,
        batch: usize,
        num_blocks: usize,
        block_elems: usize,
        perm: &[usize],
    ) -> Result<DeviceHandle> {
        let src = self.readback(x)?;
        let total = batch * num_blocks * block_elems;
        if src.len() != total || perm.len() != num_blocks {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: src.len(),
                shape: vec![batch, num_blocks, block_elems],
                size: total,
            });
        }
        let mut out = vec![0.0f32; total];
        for b in 0..batch {
            for (i, &from) in perm.iter().enumerate() {
                let d = (b * num_blocks + i) * block_elems;
                let s = (b * num_blocks + from) * block_elems;
                out[d..d + block_elems].copy_from_slice(&src[s..s + block_elems]);
            }
        }
        self.upload(&out, &[total])
    }

    /// Add `upstream` into `dest`'s slice region. Unlike `write_slice_device`,
    /// which stores, this keeps whatever the destination already held — the
    /// gradient of a tensor with more than one consumer.
    fn accumulate_slice_device(
        &self,
        dest: &DeviceHandle,
        upstream: &DeviceHandle,
        input_shape: &[usize],
        starts: &[usize],
        ends: &[usize],
    ) -> Result<DeviceHandle> {
        let mut dest_host = self.readback(dest)?;
        let upstream_host = self.readback(upstream)?;
        let expected_shape = validate_slice_shape(input_shape, starts, ends)?;
        if dest_host.len() != shape_size(input_shape)
            || upstream_host.len() != shape_size(&expected_shape)
        {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: dest_host.len().min(upstream_host.len()),
                shape: input_shape.to_vec(),
                size: shape_size(input_shape),
            });
        }
        let strides = broadcast_strides(input_shape);
        for (out_index, &value) in upstream_host.iter().enumerate() {
            let coords = linear_to_coords(out_index, &expected_shape);
            let input_index: usize = coords
                .iter()
                .enumerate()
                .map(|(axis, &coord)| (coord + starts[axis]) * strides[axis])
                .sum();
            dest_host[input_index] += value;
        }
        self.upload(&dest_host, input_shape)
    }

    fn write_slice_device(
        &self,
        dest: &DeviceHandle,
        upstream: &DeviceHandle,
        input_shape: &[usize],
        starts: &[usize],
        ends: &[usize],
    ) -> Result<DeviceHandle> {
        let mut dest_host = self.readback(dest)?;
        let upstream_host = self.readback(upstream)?;
        let expected_shape = validate_slice_shape(input_shape, starts, ends)?;
        let input_size = shape_size(input_shape);
        let expected_size = shape_size(&expected_shape);
        if dest_host.len() != input_size || upstream_host.len() != expected_size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: dest_host.len().min(upstream_host.len()),
                shape: input_shape.to_vec(),
                size: input_size,
            });
        }
        let strides = broadcast_strides(input_shape);
        for (out_index, &value) in upstream_host.iter().enumerate() {
            let out_coords = linear_to_coords(out_index, &expected_shape);
            let input_index: usize = out_coords
                .iter()
                .enumerate()
                .map(|(axis, &coord)| (coord + starts[axis]) * strides[axis])
                .sum();
            dest_host[input_index] = value;
        }
        self.upload(&dest_host, input_shape)
    }

    /// In-place AdamW step for a single parameter given host-resident
    /// gradient `grad` and device-resident `param` / `m` / `v` handles.
    ///
    /// Returns the updated `(param, m, v)` device handles. The caller owns
    /// installing them back into its store (`TensorStore::replace_device_handle`
    /// + its own moment map). Shape / length invariants:
    ///
    /// - `grad.len() == product(shape)` — the caller typically sources `grad`
    ///   via `store.to_host(grad_id)`; `matmul_backward` currently returns
    ///   host `Vec<f32>`, so keeping `grad` host avoids an upload-then-readback
    ///   round-trip just to land in this op.
    /// - `param` / `m` / `v` must already be device-resident and share `shape`.
    /// - `bc1` / `bc2` are the Adam bias-correction denominators
    ///   `1 - beta1^step` / `1 - beta2^step`, passed in so this op never sees
    ///   the step counter (matches how CUDA AdamW kernels are usually driven).
    ///
    /// This is CPU-correct by construction and gives non-Metal backends a
    /// working fallback. Metal overrides to compose the update into its lazy
    /// MLX graph so `m` / `v` / `param` stay device-resident across steps —
    /// killing the ~200-param × param-size re-upload churn that the prior
    /// `get_mut`-triggered `Dirty::Host` path caused on Qwen3.5-class models
    /// (see `docs/experience/wins/2026-04-21-adamw-on-device-metal.md`).
    ///
    /// **Eval contract:** implementations MUST return the updated
    /// handles *unevaluated*. The caller (`AdamW::step_device`) collects every
    /// param's `(new_param, new_m, new_v)` triple and fires a single
    /// `backend.eval(&handles)` at the end of the optimizer step. This turns
    /// the per-step eval count from `num_params` (~200 on Qwen3.5) into `1`
    /// regardless of parameter count — the independent per-param MLX chains
    /// share no sub-node, so batching them into one eval is safe. Backends
    /// whose `eval` is a no-op (CPU default) silently get the old semantics
    /// (work already done during the formula); only lazy-graph backends
    /// (Metal) benefit from the batching, and they MUST NOT call
    /// `mlx_eval` inside this method.
    #[allow(clippy::too_many_arguments)]
    fn adamw_step(
        &self,
        param: &DeviceHandle,
        m: &DeviceHandle,
        v: &DeviceHandle,
        grad: &[f32],
        shape: &[usize],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        bc1: f32,
        bc2: f32,
    ) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
        let size = shape_size(shape);
        if grad.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: grad.len(),
                shape: shape.to_vec(),
                size,
            });
        }
        let mut param_host = self.readback(param)?;
        let mut m_host = self.readback(m)?;
        let mut v_host = self.readback(v)?;
        if param_host.len() != size || m_host.len() != size || v_host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: param_host.len().min(m_host.len()).min(v_host.len()),
                shape: shape.to_vec(),
                size,
            });
        }
        cpu_adamw_step_in_place(
            &mut param_host,
            &mut m_host,
            &mut v_host,
            grad,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            bc1,
            bc2,
        );
        let new_param = self.upload(&param_host, shape)?;
        let new_m = self.upload(&m_host, shape)?;
        let new_v = self.upload(&v_host, shape)?;
        Ok((new_param, new_m, new_v))
    }

    /// Accepts the gradient as a `DeviceHandle` so device-resident backward
    /// ops (`embedding_backward_device`, `add_broadcast_backward_device`,
    /// `matmul_backward_device`, ...) skip the per-param `to_host(grad_id)`
    /// DtoH that otherwise makes the device-resident embedding/add_broadcast
    /// grads a +1.8% wash (and adds 41.5 GB DtoH / step).
    ///
    /// Same semantics + eval contract as `adamw_step`: returns the updated
    /// `(param, m, v)` *unevaluated* so the caller batches a single terminal
    /// `backend.eval(...)` per optimizer step. CUDA overrides to keep the
    /// gradient on-device and reuse the existing fused `adamw_step_f32`
    /// NVRTC kernel.
    #[allow(clippy::too_many_arguments)]
    fn adamw_step_device(
        &self,
        param: &DeviceHandle,
        m: &DeviceHandle,
        v: &DeviceHandle,
        grad: &DeviceHandle,
        shape: &[usize],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        bc1: f32,
        bc2: f32,
    ) -> Result<(DeviceHandle, DeviceHandle, DeviceHandle)> {
        let grad_host = self.readback(grad)?;
        self.adamw_step(
            param, m, v, &grad_host, shape, lr, beta1, beta2, eps, wd, bc1, bc2,
        )
    }

    /// Scatter-add rows into a `[vocab, feature_dim]` output.
    ///
    /// `upstream` is `[prefix_rows * feature_dim]` row-major. For each prefix
    /// position `row`, `upstream[row * feature_dim .. (row+1) * feature_dim]`
    /// is summed into `out[indices[row] * feature_dim .. (indices[row]+1) * feature_dim]`.
    /// Out-of-range or negative indices are skipped (matches the CPU/CUDA
    /// scatter-add semantics used by `embedding_backward` and
    /// `gather_last_dim_backward`). Covers both shapes:
    ///
    /// - `embedding_backward`: `feature_dim = hidden`, `vocab = weight_shape[0]`.
    /// - `gather_last_dim_backward`: `feature_dim = 1`, `vocab = src_shape.last()`.
    fn scatter_add_rows_forward(
        &self,
        upstream: &[f32],
        prefix_rows: usize,
        feature_dim: usize,
        indices: &[i32],
        vocab: usize,
    ) -> Result<Vec<f32>> {
        cpu_scatter_add_rows_forward(upstream, prefix_rows, feature_dim, indices, vocab)
    }

    /// Scatter-adds the per-token-position upstream gradient
    /// `[1, n_ids, hidden]` (or any rank that flattens to `n_ids * hidden`)
    /// into the `[vocab, hidden]` embedding table gradient. `atomicAdd` is
    /// mandatory — duplicate token ids within a single batch are normal
    /// (e.g. `the` appears N times in a 1024-token sequence) and must
    /// accumulate correctly.
    ///
    /// CUDA overrides with an NVRTC kernel that initializes the
    /// `[vocab, hidden]` output to zero and accumulates each `(b*S + s)`-row
    /// of upstream into `out[ids[b*S+s], :]` via `atomicAdd`.
    ///
    /// Keeps the embedding backward off the host so the `[B, S, H]` upstream
    /// tensor — second largest per-step DtoH in the P3.1 residue — never
    /// crosses PCIe. Hand-written because candle's `scatter_add` deliberately
    /// omits atomics.
    fn embedding_backward_device(
        &self,
        upstream_grad: &DeviceHandle,
        indices: &[i32],
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<DeviceHandle> {
        let n_ids = indices.len();
        let upstream_host = self.readback(upstream_grad)?;
        let grad =
            self.scatter_add_rows_forward(&upstream_host, n_ids, hidden_dim, indices, vocab_size)?;
        self.upload(&grad, &[vocab_size, hidden_dim])
    }

    /// Given the forward `out = a + broadcast(b, a_shape)`, this returns
    /// `grad_b = sum_over_broadcast_axes(upstream)` with output shape
    /// `b_shape`. Axes whose `b_shape` dim is 1 (or are absent from `b_shape`
    /// via right-alignment) are reduced over; matching axes pass through.
    ///
    /// `upstream` has shape `a_shape` (the output shape of the forward). The
    /// reduction is implemented as one block per output element — each block
    /// strides through the broadcast-source slots and shared-memory-reduces.
    ///
    /// CUDA overrides with an NVRTC kernel.
    fn add_broadcast_backward_device(
        &self,
        upstream: &DeviceHandle,
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Result<DeviceHandle> {
        validate_broadcast(a_shape, b_shape)?;
        let upstream_host = self.readback(upstream)?;
        let out_total: usize = if a_shape.is_empty() {
            1
        } else {
            a_shape.iter().product()
        };
        if upstream_host.len() != out_total {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: upstream_host.len(),
                shape: a_shape.to_vec(),
                size: out_total,
            });
        }
        let b_size: usize = if b_shape.is_empty() {
            1
        } else {
            b_shape.iter().product()
        };
        let mut grad_b = vec![0.0_f32; b_size];
        for (out_index, value) in upstream_host.iter().enumerate() {
            let offset = broadcast_offset(out_index, a_shape, b_shape);
            grad_b[offset] += *value;
        }
        self.upload(&grad_b, b_shape)
    }

    /// Elementwise `grad_x[i] = upstream[i] * silu'(x[i])` where
    /// `silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))`. The saved
    /// context is the original input `x` (not the output), matching the
    /// host `silu_backward`.
    ///
    /// CUDA overrides with a 1D NVRTC kernel. Returned handle is
    /// unevaluated per the batched-eval contract.
    fn silu_backward_device(
        &self,
        upstream: &DeviceHandle,
        x: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        let upstream_host = self.readback(upstream)?;
        let x_host = self.readback(x)?;
        let size = shape_size(shape);
        if upstream_host.len() != size || x_host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: upstream_host.len().min(x_host.len()),
                shape: shape.to_vec(),
                size,
            });
        }
        let grad: Vec<f32> = x_host
            .iter()
            .zip(upstream_host.iter())
            .map(|(&xv, &up)| {
                let sigmoid = 1.0 / (1.0 + (-xv).exp());
                let deriv = sigmoid + (xv * sigmoid * (1.0 - sigmoid));
                up * deriv
            })
            .collect();
        self.upload(&grad, shape)
    }

    /// Elementwise `grad_x[i] = upstream[i] * gelu'(x[i])` where
    /// `gelu'(x) = 0.5*(1 + erf(x/√2)) + x * (1/√(2π)) * exp(-x²/2)`.
    /// (erf form, matches the autograd `gelu_host_eager` forward.)
    ///
    /// CUDA overrides with a 1D NVRTC kernel. Returned handle is unevaluated.
    fn gelu_backward_device(
        &self,
        upstream: &DeviceHandle,
        x: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        const INV_SQRT_2: f32 = 0.707_106_77;
        const INV_SQRT_2PI: f32 = 0.398_942_3;
        let upstream_host = self.readback(upstream)?;
        let x_host = self.readback(x)?;
        let size = shape_size(shape);
        if upstream_host.len() != size || x_host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: upstream_host.len().min(x_host.len()),
                shape: shape.to_vec(),
                size,
            });
        }
        let grad: Vec<f32> = x_host
            .iter()
            .zip(upstream_host.iter())
            .map(|(&xv, &up)| {
                let erf_term = libm::erff(xv * INV_SQRT_2);
                let exp_term = (-0.5 * xv * xv).exp();
                let deriv = 0.5 * (1.0 + erf_term) + (xv * INV_SQRT_2PI * exp_term);
                up * deriv
            })
            .collect();
        self.upload(&grad, shape)
    }

    /// Consumes the saved output `y`:
    /// `grad_x[i] = upstream[i] * y[i] * (1 - y[i])`.
    ///
    /// CUDA overrides with a 1D NVRTC kernel.
    fn sigmoid_backward_device(
        &self,
        upstream: &DeviceHandle,
        y: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        let upstream_host = self.readback(upstream)?;
        let y_host = self.readback(y)?;
        let size = shape_size(shape);
        if upstream_host.len() != size || y_host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: upstream_host.len().min(y_host.len()),
                shape: shape.to_vec(),
                size,
            });
        }
        let grad: Vec<f32> = y_host
            .iter()
            .zip(upstream_host.iter())
            .map(|(&yv, &up)| up * yv * (1.0 - yv))
            .collect();
        self.upload(&grad, shape)
    }

    /// Consumes the saved input `x`:
    /// `grad_x[i] = upstream[i] * sign(x[i])`, with `sign(0) = 0`.
    ///
    /// CUDA overrides with a 1D NVRTC kernel.
    fn abs_backward_device(
        &self,
        upstream: &DeviceHandle,
        x: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        let upstream_host = self.readback(upstream)?;
        let x_host = self.readback(x)?;
        let size = shape_size(shape);
        if upstream_host.len() != size || x_host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: upstream_host.len().min(x_host.len()),
                shape: shape.to_vec(),
                size,
            });
        }
        let grad: Vec<f32> = x_host
            .iter()
            .zip(upstream_host.iter())
            .map(|(&xv, &up)| up * cpu_sign(xv))
            .collect();
        self.upload(&grad, shape)
    }

    /// Consumes the saved output `y = exp(x)`:
    /// `grad_x[i] = upstream[i] * y[i]`.
    ///
    /// CUDA overrides with a 1D NVRTC kernel.
    fn exp_backward_device(
        &self,
        upstream: &DeviceHandle,
        y: &DeviceHandle,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        let upstream_host = self.readback(upstream)?;
        let y_host = self.readback(y)?;
        let size = shape_size(shape);
        if upstream_host.len() != size || y_host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: upstream_host.len().min(y_host.len()),
                shape: shape.to_vec(),
                size,
            });
        }
        let grad: Vec<f32> = y_host
            .iter()
            .zip(upstream_host.iter())
            .map(|(&yv, &up)| up * yv)
            .collect();
        self.upload(&grad, shape)
    }

    /// Returns `(grad_a, grad_b)` where `grad_a[i] = upstream[i] * b[i]`
    /// and `grad_b[i] = upstream[i] * a[i]`. `need_grad_a` / `need_grad_b`
    /// short-circuit each side to `None` (mirrors `matmul_backward_device`).
    ///
    /// CUDA overrides with two 1D NVRTC kernels.
    fn mul_backward_device(
        &self,
        upstream: &DeviceHandle,
        a: &DeviceHandle,
        b: &DeviceHandle,
        shape: &[usize],
        need_grad_a: bool,
        need_grad_b: bool,
    ) -> Result<(Option<DeviceHandle>, Option<DeviceHandle>)> {
        if !need_grad_a && !need_grad_b {
            return Ok((None, None));
        }
        let upstream_host = self.readback(upstream)?;
        let a_host = if need_grad_b {
            Some(self.readback(a)?)
        } else {
            None
        };
        let b_host = if need_grad_a {
            Some(self.readback(b)?)
        } else {
            None
        };
        let size = shape_size(shape);
        if upstream_host.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: upstream_host.len(),
                shape: shape.to_vec(),
                size,
            });
        }
        let grad_a = if need_grad_a {
            let b = b_host.as_ref().expect("requested above");
            let grad: Vec<f32> = upstream_host
                .iter()
                .zip(b.iter())
                .map(|(&up, &bv)| up * bv)
                .collect();
            Some(self.upload(&grad, shape)?)
        } else {
            None
        };
        let grad_b = if need_grad_b {
            let a = a_host.as_ref().expect("requested above");
            let grad: Vec<f32> = upstream_host
                .iter()
                .zip(a.iter())
                .map(|(&up, &av)| up * av)
                .collect();
            Some(self.upload(&grad, shape)?)
        } else {
            None
        };
        Ok((grad_a, grad_b))
    }

    /// Returns `(grad_x, grad_w)` where each side is gated by the
    /// corresponding `need_grad_*` flag (default impl skips the host
    /// allocation for skipped sides).
    ///
    /// Math (mirrors `cpu_rmsnorm_backward`):
    ///   inv_rms[r] = 1 / sqrt(mean(x[r,:]^2) + eps)
    ///   dot[r]     = sum_j(upstream[r,j] * weight[j] * x[r,j])
    ///   grad_x[r,j] = inv*upstream[r,j]*weight[j] - x[r,j]*inv*inv*dot/H
    ///   grad_w[j]   = sum_r(upstream[r,j] * x[r,j] * inv_rms[r])
    ///
    /// CUDA overrides with three NVRTC kernels (per-row inv_rms scratch,
    /// then per-row grad_x with shared-mem `dot` reduce, then per-col
    /// grad_w reduce).
    fn rms_norm_backward_device(
        &self,
        upstream: &DeviceHandle,
        x: &DeviceHandle,
        weight: &DeviceHandle,
        shape: &[usize],
        eps: f32,
        need_grad_x: bool,
        need_grad_w: bool,
    ) -> Result<(Option<DeviceHandle>, Option<DeviceHandle>)> {
        if !need_grad_x && !need_grad_w {
            return Ok((None, None));
        }
        let hidden = *shape.last().ok_or(crate::AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        })?;
        let total = shape_size(shape);
        let rows = total.checked_div(hidden).unwrap_or(0);

        let upstream_host = self.readback(upstream)?;
        let x_host = self.readback(x)?;
        let weight_host = self.readback(weight)?;
        if upstream_host.len() != total || x_host.len() != total || weight_host.len() != hidden {
            return Err(crate::AutogradError::ShapeMismatch {
                expected: shape.to_vec(),
                got: vec![upstream_host.len()],
            });
        }

        let mut inv_rms = vec![0.0_f32; rows];
        for (row, inv_slot) in inv_rms.iter_mut().enumerate() {
            let base = row * hidden;
            let mut sum_sq = 0.0_f32;
            for col in 0..hidden {
                let v = x_host[base + col];
                sum_sq += v * v;
            }
            *inv_slot = 1.0 / ((sum_sq / hidden as f32) + eps).sqrt();
        }

        let grad_x = if need_grad_x {
            let mut grad = vec![0.0_f32; total];
            for (row, &inv) in inv_rms.iter().enumerate() {
                let base = row * hidden;
                let mut dot = 0.0_f32;
                for col in 0..hidden {
                    dot += upstream_host[base + col] * weight_host[col] * x_host[base + col];
                }
                let correction = inv * inv * dot / hidden as f32;
                for col in 0..hidden {
                    grad[base + col] = (inv * upstream_host[base + col] * weight_host[col])
                        - (x_host[base + col] * inv * correction);
                }
            }
            Some(self.upload(&grad, shape)?)
        } else {
            None
        };
        let grad_w = if need_grad_w {
            let mut grad = vec![0.0_f32; hidden];
            for (row, &inv) in inv_rms.iter().enumerate() {
                let base = row * hidden;
                for col in 0..hidden {
                    grad[col] += upstream_host[base + col] * x_host[base + col] * inv;
                }
            }
            Some(self.upload(&grad, &[hidden])?)
        } else {
            None
        };
        Ok((grad_x, grad_w))
    }

    /// The backward is identical to the forward with `sin` negated:
    ///   grad_x = rope_forward(upstream, cos, -sin)
    ///
    /// `cos`/`sin` stay host-side (mirrors `rope` forward — caches are
    /// per-seq and seldom benefit from being device-resident).
    ///
    /// CUDA overrides with a dedicated NVRTC kernel that inlines the sign
    /// flip.
    fn rope_backward_device(
        &self,
        upstream: &DeviceHandle,
        x_shape: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<DeviceHandle> {
        let upstream_host = self.readback(upstream)?;
        let neg_sin: Vec<f32> = sin.iter().map(|&v| -v).collect();
        let grad = cpu_rope_forward(&upstream_host, x_shape, cos, &neg_sin)?;
        self.upload(&grad, x_shape)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn upload(&self, host: &[f32], _shape: &[usize]) -> Result<DeviceHandle> {
        Ok(DeviceHandle::Cpu(host.to_vec()))
    }

    fn readback(&self, handle: &DeviceHandle) -> Result<Vec<f32>> {
        match handle {
            DeviceHandle::Cpu(data) => Ok(data.clone()),
            #[cfg(feature = "metal")]
            DeviceHandle::Metal(_) => Err(crate::AutogradError::TapeInvariant(
                "cpu backend cannot read back a metal device handle",
            )),
            #[cfg(feature = "cuda")]
            DeviceHandle::Cuda(_) => Err(crate::AutogradError::TapeInvariant(
                "cpu backend cannot read back a cuda device handle",
            )),
            #[cfg(feature = "cuda")]
            DeviceHandle::CudaBf16(_) => Err(crate::AutogradError::TapeInvariant(
                "cpu backend cannot read back a cuda bf16 device handle",
            )),
            #[cfg(feature = "cuda")]
            DeviceHandle::CudaFp8BlockScaled(_) => Err(crate::AutogradError::TapeInvariant(
                "cpu backend cannot read back a cuda fp8 block-scaled device handle",
            )),
            #[cfg(feature = "cuda")]
            DeviceHandle::CudaFp4E2M1Group(_) => Err(crate::AutogradError::TapeInvariant(
                "cpu backend cannot read back a cuda nvfp4 device handle",
            )),
        }
    }

    fn eval(&self, _handles: &[&DeviceHandle]) -> Result<()> {
        Ok(())
    }

    #[allow(irrefutable_let_patterns)]
    fn matmul(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        let DeviceHandle::Cpu(a_data) = a else {
            return Err(crate::AutogradError::TapeInvariant(
                "cpu backend cannot matmul a non-cpu device handle",
            ));
        };
        let DeviceHandle::Cpu(b_data) = b else {
            return Err(crate::AutogradError::TapeInvariant(
                "cpu backend cannot matmul a non-cpu device handle",
            ));
        };
        let (out, out_shape) = cpu_matmul_forward(a_data, a_shape, b_data, b_shape)?;
        Ok((DeviceHandle::Cpu(out), out_shape))
    }

    fn matmul_forward(
        &self,
        a: &[f32],
        a_shape: &[usize],
        b: &[f32],
        b_shape: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        cpu_matmul_forward(a, a_shape, b, b_shape)
    }

    #[allow(irrefutable_let_patterns)]
    fn matmul_bt(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        let DeviceHandle::Cpu(a_data) = a else {
            return Err(crate::AutogradError::TapeInvariant(
                "cpu backend cannot matmul_bt a non-cpu device handle",
            ));
        };
        let DeviceHandle::Cpu(b_data) = b else {
            return Err(crate::AutogradError::TapeInvariant(
                "cpu backend cannot matmul_bt a non-cpu device handle",
            ));
        };
        let (out, out_shape) = cpu_matmul_bt_forward(a_data, a_shape, b_data, b_shape)?;
        Ok((DeviceHandle::Cpu(out), out_shape))
    }

    #[allow(irrefutable_let_patterns)]
    fn add(&self, a: &DeviceHandle, b: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let DeviceHandle::Cpu(a_data) = a else {
            return Err(crate::AutogradError::TapeInvariant(
                "cpu backend cannot add a non-cpu device handle",
            ));
        };
        let DeviceHandle::Cpu(b_data) = b else {
            return Err(crate::AutogradError::TapeInvariant(
                "cpu backend cannot add a non-cpu device handle",
            ));
        };
        let size = shape_size(shape);
        if a_data.len() != size || b_data.len() != size {
            return Err(crate::AutogradError::ShapeMismatch {
                expected: vec![size],
                got: vec![a_data.len().min(b_data.len())],
            });
        }
        let out: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(lhs, rhs)| lhs + rhs)
            .collect();
        Ok(DeviceHandle::Cpu(out))
    }

    #[allow(irrefutable_let_patterns)]
    fn sum_all(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let DeviceHandle::Cpu(data) = x else {
            return Err(crate::AutogradError::TapeInvariant(
                "cpu backend cannot sum a non-cpu device handle",
            ));
        };
        let size = shape_size(shape);
        if data.len() != size {
            return Err(crate::AutogradError::DataLengthMismatch {
                len: data.len(),
                shape: shape.to_vec(),
                size,
            });
        }
        let total: f32 = data.iter().sum();
        Ok(DeviceHandle::Cpu(vec![total]))
    }
}

#[path = "backend/cpu_math.rs"]
mod cpu_math;
pub use cpu_math::*;
// Re-export pub(crate) items from cpu_math so sibling modules (backend_cuda,
// backend_metal) can import them via `crate::backend::*`.
#[allow(unused_imports)]
pub(crate) use cpu_math::{
    broadcast_offset, broadcast_strides, cpu_qwen_decode_prepare_kv, cpu_qwen_decode_prepare_q,
    linear_to_coords, matmul_output_shape, shape_size, validate_broadcast,
    validate_qwen_decode_prepare_kv_shapes, validate_qwen_decode_prepare_q_shapes,
    validate_slice_shape,
};
