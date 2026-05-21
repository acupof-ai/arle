//! Backend abstraction for heavy ops. Today: matmul forward only.
//!
//! Transformer training is ~90% matmul FLOPs; moving matmul to GPU swings the
//! big lever without requiring device-resident tensors. Host `Vec<f32>`
//! stays authoritative; GPU backends upload, compute, and download per
//! call. Non-matmul ops (softmax, elementwise, norm, gather) stay on CPU.
//!
//! The trait is additive — future ops land as new methods with CPU
//! fallbacks so a backend does not need to implement every op day one.

use crate::Result;
#[cfg(any(feature = "metal", feature = "cuda"))]
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Metal,
    Cuda,
}

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
}

#[derive(Debug, Clone)]
pub enum DeviceHandle {
    Cpu(Vec<f32>),
    #[cfg(feature = "metal")]
    Metal(MlxHandle),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
}

type QwenDecodePrepareQHost = (Vec<f32>, Option<Vec<f32>>, Vec<usize>);
type QwenDecodePrepareKvHost = (Vec<f32>, Vec<f32>, Vec<usize>);

#[derive(Debug, Clone)]
pub struct DeviceGradClipResult {
    pub pre_clip_norm: f64,
    pub clipped_grads: Option<Vec<DeviceHandle>>,
}

pub trait Backend: std::fmt::Debug + Send + Sync {
    fn device(&self) -> Device;

    fn upload(&self, host: &[f32], _shape: &[usize]) -> Result<DeviceHandle> {
        Ok(DeviceHandle::Cpu(host.to_vec()))
    }

    fn import_bf16_device_ptr_as_f32(
        &self,
        src_device_ptr: u64,
        len: usize,
        shape: &[usize],
    ) -> Result<DeviceHandle> {
        let _ = (src_device_ptr, len, shape);
        Err(crate::AutogradError::TapeInvariant(
            "backend does not support importing bf16 device pointers",
        ))
    }

    /// Allocate a zero-filled device handle for `shape`.
    ///
    /// Default implementation uploads a host zero vector so existing
    /// backends inherit correct behavior. CUDA overrides to allocate and
    /// memset on device, which avoids first-step AdamW moment HtoD traffic.
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
        }
    }

    fn eval(&self, _handles: &[&DeviceHandle]) -> Result<()> {
        Ok(())
    }

    /// Whether `Tape::backward` should `flush_to_host_batch` every
    /// device-resident tape output **before** walking backward. Metal
    /// returns `true` because each `mlx_eval` round-trip dominates at
    /// small shapes and batching N FFI guards into 1 is a real win.
    /// CUDA returns `false` (default) — the batch readback there is the
    /// 1 GB DtoH the M5.3b / Wave 1 / P1 / P2 / P3 milestones could
    /// never kill, and per-op lazy readback is strictly cheaper because
    /// device-resident downstream backward ops never need the host
    /// snapshot in the first place. See
    /// `docs/research/2026-05-17-cuda-training-architectural-correction.md`
    /// and the P3 wins entry.
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
    /// Default fallback reads host buffers and calls `cpu_matmul_bt_forward`.
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
    /// Semantics: `grad_a = dC @ B^T` and `grad_b = A^T @ dC`. The default
    /// implementation forwards to `cpu_matmul_backward`; Metal/CUDA override
    /// to run on-device.
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

    /// Device-handle variant of `matmul_backward`. Foundation for the
    /// device-resident gradient tape — see
    /// `docs/research/2026-05-17-cuda-training-architectural-correction.md`.
    ///
    /// Computes `grad_a = grad_out @ B^T` and `grad_b = A^T @ grad_out` and
    /// returns each as an *unevaluated* `DeviceHandle` so the caller can
    /// batch a single terminal `backend.eval(...)` per training step (mirrors
    /// the M5.3b.11 contract used by `adamw_step` /
    /// `log_softmax_last_axis_backward`). `need_grad_a` / `need_grad_b`
    /// short-circuit to `None` so the unused SGEMM is never launched.
    ///
    /// The default implementation does
    /// `readback → cpu_matmul_backward → upload` so non-CUDA backends silently
    /// inherit correct (but slow) behaviour; CUDA overrides to keep both
    /// SGEMMs on-device with no host roundtrip.
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

    /// Device-handle backward for `C = A @ B^T`. Default fallback uses host
    /// buffers and uploads the requested gradients.
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

    /// Elementwise `C = A + B` over identically-shaped contiguous tensors.
    /// Lazy on backends that support it (e.g. Metal defers to `mlx_eval`).
    fn add(&self, a: &DeviceHandle, b: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle>;

    /// In-place gradient accumulation: returns a fresh `DeviceHandle`
    /// holding `dest + src` elementwise. Foundation for the device-resident
    /// gradient tape — when two backward paths converge on the same
    /// parameter, the merge runs through this op rather than a host
    /// `accumulate_grad` roundtrip. See
    /// `docs/research/2026-05-17-cuda-training-architectural-correction.md`.
    ///
    /// `dest` and `src` must share `shape` and `product(shape)` elements.
    /// The returned handle is *unevaluated* on backends with a lazy graph
    /// (CUDA returns a `CudaSlice` ready for the batched `eval`).
    ///
    /// The default implementation does `readback both → host add → upload`
    /// so CPU/Metal inherit correct behaviour; CUDA overrides with a
    /// 1D NVRTC kernel.
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

    /// Sum of squares for a device handle, returned on host as `f64`.
    /// The default fallback reads the full tensor; CUDA overrides with a
    /// partial-reduction kernel so gradient clipping can stay device-resident.
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

    /// Device-resident global-norm gradient clip across many tensors.
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

    /// Device-handle variant of `softmax_forward_last_axis`. Lazy on backends
    /// that can compose softmax into their graph (Metal: `mlx_softmax_axis`);
    /// the default implementation falls back to `readback → host compute →
    /// upload` so CPU/CUDA need no special-case. M5.3b.2.
    fn softmax_last_axis(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.softmax_forward_last_axis(&host, shape)?;
        self.upload(&out, shape)
    }

    /// Device-handle variant of `log_softmax_forward_last_axis`. Lazy on
    /// backends that can compose into their graph (Metal uses
    /// `mlx_logsumexp_axis` + `mlx_subtract`); the default implementation
    /// falls back to `readback → host compute → upload`. M5.3b.2.
    fn log_softmax_last_axis(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.log_softmax_forward_last_axis(&host, shape)?;
        self.upload(&out, shape)
    }

    /// Device-handle variant of softmax backward. Computes
    /// `grad_input = y * (upstream - sum(upstream * y, axis=-1, keepdim=true))`
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

    /// Device-handle variant of `cpu_log_softmax_backward`. Computes
    /// `grad_input = upstream - exp(log_softmax_output) * sum(upstream, axis=-1, keepdim=true)`
    /// row-wise over the last axis.
    ///
    /// `log_softmax_output` is the saved forward output (NOT the input —
    /// `softmax(x) = exp(log_softmax(x))` and the backward identity uses the
    /// softmax probability, which is just `exp(saved_output)`). `upstream`
    /// has the same shape as `log_softmax_output`.
    ///
    /// Wave 1 (post-M5.3b-nsys attribution): the default fallback runs
    /// the host formula via `cpu_log_softmax_backward`, so non-CUDA
    /// backends inherit correct behaviour. CUDA overrides this with a
    /// single per-row NVRTC kernel that consumes the saved forward output
    /// without a host roundtrip — kills the `[B, S, V]` × 4 B ≈ 1 GB DtoH
    /// copy that nsys identified as the single largest readback per
    /// training step (see
    /// `docs/research/2026-05-17-cuda-training-step-nsys-attribution.md`).
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

    /// Device-handle variant of `silu_forward`. Lazy on backends that can
    /// compose `x * sigmoid(x)` into their graph (Metal: `mlx_multiply` +
    /// `mlx_sigmoid`); the default implementation falls back to
    /// `readback → host compute → upload` so CPU/CUDA need no special-case.
    /// M5.3b.3.
    fn silu(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.silu_forward(&host)?;
        self.upload(&out, shape)
    }

    /// Device-handle variant of `exp_forward`. Lazy on backends with a
    /// native `exp` graph node (Metal: `mlx_exp`); the default
    /// implementation falls back to `readback → host compute → upload`
    /// so CPU/CUDA need no special-case. M5.3b.4.
    fn exp(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.exp_forward(&host)?;
        self.upload(&out, shape)
    }

    /// Elementwise `out = 1 / (1 + exp(-a))`.
    fn sigmoid_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        cpu_sigmoid_forward(a)
    }

    /// Device-handle variant of `sigmoid_forward`. Lazy on backends with a
    /// native `sigmoid` graph node (Metal: `mlx_sigmoid`); the default
    /// implementation falls back to `readback → host compute → upload`
    /// so CPU/CUDA need no special-case. M5.3b.18.
    fn sigmoid(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.sigmoid_forward(&host)?;
        self.upload(&out, shape)
    }

    /// Elementwise `out = a * b` over identically-sized contiguous tensors.
    fn mul_forward(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        cpu_mul_forward(a, b)
    }

    /// Device-handle variant of `mul_forward`. Lazy on backends that can
    /// compose `a * b` into their graph (Metal: `mlx_multiply`); default
    /// falls back to `readback(a) → readback(b) → host compute → upload`
    /// so CPU/CUDA need no special-case. Shapes must match on both sides
    /// (elementwise, not broadcasted — use `add_broadcast`'s `mul` twin if
    /// broadcast multiplication is ever needed). M5.3b.17.
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

    /// Device-handle variant of `mul_scalar_forward`. Lazy on backends
    /// that can compose `x * s` into their graph (Metal: broadcast
    /// `mlx_multiply` against a rank-0 scalar `mlx_array`); the default
    /// implementation falls back to `readback → host compute → upload`
    /// so CPU/CUDA need no special-case. M5.3b.13.
    fn mul_scalar(&self, x: &DeviceHandle, s: f32, shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        let out = self.mul_scalar_forward(&host, s)?;
        self.upload(&out, shape)
    }

    /// P3: device-resident backward for `mul_scalar(x, k)`. Computes
    /// `grad_x[i] = upstream[i] * k` and returns an unevaluated handle.
    ///
    /// The default fallback runs `readback → host multiply → upload` so
    /// CPU/Metal inherit correct behaviour. CUDA overrides with a 1D
    /// NVRTC kernel.
    ///
    /// Wires the CE-loss backward chain that P2 / Wave 1 / M5.3b.x already
    /// device-overrode: `mul_scalar_backward` was the *first* host op in
    /// `d_loss → mul_scalar_backward → mean_backward → gather_backward →
    /// log_softmax_backward → matmul_backward`, so its host fallback
    /// demoted every downstream `device_path_ok` gate to host. Keeping
    /// this on-device unblocks the whole chain — see
    /// `docs/research/2026-05-17-cuda-training-architectural-correction.md`.
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

    /// P3: device-resident backward for `mean(x)`. The forward reduces
    /// `elem_count = product(output_shape)` elements to a rank-0 scalar;
    /// the backward broadcasts `upstream_grad / elem_count` across
    /// `elem_count` slots of the returned `d_input` handle.
    ///
    /// `upstream_grad` must be a rank-0 scalar (shape `[]` or `[1]`).
    /// `output_shape` is the shape of the input to the original `mean`
    /// op (i.e. the shape of the returned `d_input`).
    ///
    /// The default fallback runs `readback scalar → host broadcast-scale
    /// → upload` so CPU/Metal inherit correct behaviour. CUDA overrides
    /// with a 1D NVRTC kernel that fetches the upstream scalar from
    /// device memory (free L1 broadcast) and writes one slot per thread.
    ///
    /// Pairs with `mul_scalar_backward_device` to keep the CE-loss
    /// backward chain device-resident — see
    /// `docs/research/2026-05-17-cuda-training-architectural-correction.md`.
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

    /// Device-handle variant of `add_broadcast_forward`. Lazy on backends
    /// whose native add already broadcasts (Metal: `mlx_add` — NumPy-style
    /// right-aligned broadcasting, no explicit `broadcast_to` needed); the
    /// default implementation falls back to `readback → host compute →
    /// upload` so CPU/CUDA need no special-case. Output shape equals
    /// `a_shape` (the same contract as `add_broadcast_forward`). M5.3b.14.
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

    /// Device-handle variant of `rms_norm_forward`. Lazy on backends with
    /// a native fused rms-norm op (Metal: `mlx_fast_rms_norm` over a
    /// borrowed `x` handle + per-call `weight` upload); the default
    /// implementation falls back to `readback → host compute → upload`.
    /// Backward path recomputes `inv_rms` host-side — see `ops::norm`
    /// for the saved-context encoding. M5.3b.6.
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

    /// Device-handle variant of `embedding_forward`. Lazy on backends that
    /// can compose the row-gather into their eval stream (Metal: upload
    /// `ids` as a tiny int32 array + `mlx_take_axis` + reshape, no eval).
    /// Output shape is `[1, ids.len(), dim]` — matching
    /// `ops::embedding`'s convention of treating raw ids as a single batch
    /// row. Default implementation falls back to `readback → host compute →
    /// upload`. M5.3b.7.
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
    /// Default fallback reads those tiny ids to host and reuses `embedding`.
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
        let mut out = Vec::with_capacity(rows);
        for row in 0..rows {
            let base = row * vocab;
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &value) in host[base..base + vocab].iter().enumerate() {
                if value > best_val {
                    best_val = value;
                    best_idx = idx;
                }
            }
            out.push(best_idx as f32);
        }
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

    /// Device-handle lazy GELU (erf form), matching `ops::activation::gelu`'s
    /// CPU body: `0.5 * x * (1 + erf(x / sqrt(2)))`. NOT the tanh-approx
    /// variant exposed by `gelu_forward` — those two formulas differ at the
    /// ~1e-3 level, and `gelu_backward` hard-codes the erf-derivative via
    /// the saved input, so forward must stay on the erf form for the
    /// saved-input derivative to be consistent. Default implementation
    /// falls back to `readback → host erf compute → upload`. M5.3b.8.
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

    /// Device-handle variant of `rope_forward`. Lazy on backends that can
    /// compose the half-split rotation graph into their eval stream (Metal:
    /// `mlx_slice` → `mlx_multiply` → `mlx_subtract`/`mlx_add` → `mlx_concatenate_axis`,
    /// no eval). `cos`/`sin` stay as host slices — the caches are precomputed
    /// per seq length and seldom benefit from being device-resident, and
    /// keeping them host-side means no merge of device handles is required.
    /// Default implementation falls back to `readback → host compute →
    /// upload`. M5.3b.5.
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

    /// Device-handle variant of `gather_last_dim_forward`. Lazy on backends
    /// that can compose `flatten → take_axis → reshape` into their eval
    /// stream (Metal: `mlx_reshape` to `[prefix*vocab]`, `mlx_take_axis`
    /// with remapped `i * vocab + ids[i]` flat ids, `mlx_reshape` back to
    /// `src_shape[..-1]`). Default implementation falls back to
    /// `readback → host compute → upload`. Output shape is
    /// `src_shape[..-1]` (empty for rank-1 input). M5.3b.9.
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

    /// Device-handle backward for `gather_last_dim`. Zero-fills a
    /// `[prefix_rows, vocab] = src_shape` output and scatters the per-prefix
    /// `upstream` values into the `(row, ids[row])` slots. Equivalent to the
    /// flat `scatter_add_rows_forward(upstream, prefix_rows, 1,
    /// remapped_ids, prefix_rows * vocab)` path the host backward takes, but
    /// the trait-level signature exposes the natural `[B, S, V]` output
    /// shape so backends can pick block tiling (one block per prefix row of
    /// `vocab` cols) without un-flattening.
    ///
    /// `upstream` has length `product(src_shape[..-1])` (one scalar per
    /// prefix position). `indices.len() == prefix_rows == upstream.len()`.
    /// Negative or out-of-range indices are silently skipped, matching
    /// `cpu_gather_last_dim_backward` / `cpu_scatter_add_rows_forward`.
    ///
    /// Wave 1: CUDA overrides this with a single per-row NVRTC kernel so
    /// the `[B, S, V]` grad stays device-resident — keeps the `1 GB`
    /// scatter-add output off the host-roundtrip path that the host
    /// `gather_last_dim_backward` previously forced (see
    /// `docs/research/2026-05-17-cuda-training-step-nsys-attribution.md`).
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
    /// the same logical elements. Numel must match (`product(new_shape) ==
    /// product(old_shape)`); the caller is expected to have checked that.
    /// Default: readback → host (no-op reshape, contiguous) → upload.
    /// Metal overrides to `mlx_reshape` so the whole graph stays lazy —
    /// reshape is a free metadata op on MLX side. M5.3b.12.
    fn reshape(&self, x: &DeviceHandle, new_shape: &[usize]) -> Result<DeviceHandle> {
        let host = self.readback(x)?;
        self.upload(&host, new_shape)
    }

    /// Swap two axes of `x`. `old_shape` is the pre-swap shape; the caller is
    /// responsible for computing the post-swap shape (just swap the two
    /// entries). `axis1`/`axis2` must be valid axes into `old_shape`. Default:
    /// readback → host transpose loop → upload. Metal overrides to
    /// `mlx_transpose_axes` with a permutation that is identity except for
    /// the two swapped positions, composing into the lazy graph. M5.3b.12.
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
    /// logical shape is `ends - starts` (caller computes). Default: readback →
    /// host slice loop → upload. Metal overrides to `mlx_slice` with
    /// strides=1, wrapping the non-contiguous view in `mlx_contiguous` so
    /// readback respects the sliced window (same rationale as the
    /// `transpose_axes_swap` override). M5.3b.16.
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
    /// sequence axis. Default: readback → host reference → upload. CUDA
    /// overrides this for OPD rollout KV-cache appends so cached K/V stay
    /// device-resident during greedy decode.
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

    /// Device-handle backward for a contiguous slice. Returns a full
    /// `old_shape` gradient with upstream values scattered into the sliced
    /// window and zeros elsewhere.
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
    /// Default implementation: `readback(param, m, v) → host formula → upload`.
    /// This is CPU-correct by construction and gives non-Metal backends a
    /// working fallback. Metal overrides to compose the update into its lazy
    /// MLX graph so `m` / `v` / `param` stay device-resident across steps —
    /// killing the ~200-param × param-size re-upload churn that the prior
    /// `get_mut`-triggered `Dirty::Host` path caused on Qwen3.5-class models
    /// (see `docs/experience/wins/2026-04-21-adamw-on-device-metal.md`).
    ///
    /// **Eval contract (M5.3b.11):** implementations MUST return the updated
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

    /// Wave 2.0: device-grad variant of `adamw_step`. Accepts the gradient as
    /// a `DeviceHandle` so device-resident backward ops
    /// (`embedding_backward_device`, `add_broadcast_backward_device`,
    /// `matmul_backward_device`, ...) skip the per-param `to_host(grad_id)` DtoH
    /// that turned Wave 2 Commit A into a +1.8% wash (and added 41.5 GB DtoH /
    /// step). See `docs/experience/wins/2026-05-17-bench-pretrain-wave2a-embedding-addbcast.md`.
    ///
    /// Same semantics + eval contract as `adamw_step`: returns the updated
    /// `(param, m, v)` *unevaluated* so the caller batches a single terminal
    /// `backend.eval(...)` per optimizer step. Default impl: `readback(grad) →
    /// self.adamw_step(...)` so CPU/Metal silently inherit correctness through
    /// the host fallback. CUDA overrides to keep the gradient on-device and
    /// reuse the existing fused `adamw_step_f32` NVRTC kernel.
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

    /// Wave 2 Commit A: device-resident backward for `embedding`. Scatter-adds
    /// the per-token-position upstream gradient `[1, n_ids, hidden]` (or any
    /// rank that flattens to `n_ids * hidden`) into the `[vocab, hidden]`
    /// embedding table gradient. `atomicAdd` is mandatory — duplicate token
    /// ids within a single batch are normal (e.g. `the` appears N times in a
    /// 1024-token sequence) and must accumulate correctly.
    ///
    /// Default fallback runs `readback → cpu_scatter_add_rows_forward →
    /// upload` so CPU/Metal inherit correct behaviour. CUDA overrides with
    /// an NVRTC kernel that initializes the `[vocab, hidden]` output to zero
    /// and accumulates each `(b*S + s)`-row of upstream into
    /// `out[ids[b*S+s], :]` via `atomicAdd`.
    ///
    /// Keeps the embedding backward off the host so the `[B, S, H]` upstream
    /// tensor — second largest per-step DtoH in the P3.1 residue — never
    /// crosses PCIe. See
    /// `docs/research/2026-05-17-candle-kernel-vendor-survey.md` §1 for why
    /// hand-write (candle's `scatter_add` deliberately omits atomics).
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

    /// Wave 2 Commit A: device-resident backward for `add_broadcast`.
    /// Given the forward `out = a + broadcast(b, a_shape)`, this returns
    /// `grad_b = sum_over_broadcast_axes(upstream)` with output shape
    /// `b_shape`. Axes whose `b_shape` dim is 1 (or are absent from `b_shape`
    /// via right-alignment) are reduced over; matching axes pass through.
    ///
    /// `upstream` has shape `a_shape` (the output shape of the forward). The
    /// reduction is implemented as one block per output element — each block
    /// strides through the broadcast-source slots and shared-memory-reduces.
    ///
    /// Default fallback runs `readback → host loop (mirrors
    /// `add_broadcast_backward` host path) → upload`. CUDA overrides with an
    /// NVRTC kernel.
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

    /// Wave 2.1: device-resident backward for `silu(x)`. Elementwise
    /// `grad_x[i] = upstream[i] * silu'(x[i])` where
    /// `silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))`. The saved
    /// context is the original input `x` (not the output), matching the
    /// host `silu_backward`.
    ///
    /// Default fallback: `readback(upstream) → readback(x) → host loop →
    /// upload`. CUDA overrides with a 1D NVRTC kernel. Returned handle is
    /// unevaluated per the M5.3b.11 batched-eval contract.
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

    /// Wave 2.1: device-resident backward for `gelu(x)` (erf form, matches
    /// the autograd `gelu_host_eager` forward). Elementwise
    /// `grad_x[i] = upstream[i] * gelu'(x[i])` where
    /// `gelu'(x) = 0.5*(1 + erf(x/√2)) + x * (1/√(2π)) * exp(-x²/2)`.
    ///
    /// Default fallback: `readback → host loop → upload`. CUDA overrides
    /// with a 1D NVRTC kernel. Returned handle is unevaluated.
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

    /// Wave 2.1: device-resident backward for `sigmoid(x)`. Consumes the
    /// saved output `y`: `grad_x[i] = upstream[i] * y[i] * (1 - y[i])`.
    ///
    /// Default fallback: `readback → host loop → upload`. CUDA overrides
    /// with a 1D NVRTC kernel.
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

    /// Wave 2.1: device-resident backward for `exp(x)`. Consumes the saved
    /// output `y = exp(x)`: `grad_x[i] = upstream[i] * y[i]`.
    ///
    /// Default fallback: `readback → host multiply → upload`. CUDA
    /// overrides with a 1D NVRTC kernel.
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

    /// Wave 2.1: device-resident backward for `mul(a, b)`. Returns
    /// `(grad_a, grad_b)` where `grad_a[i] = upstream[i] * b[i]` and
    /// `grad_b[i] = upstream[i] * a[i]`. `need_grad_a` / `need_grad_b`
    /// short-circuit each side to `None` (mirrors `matmul_backward_device`).
    ///
    /// Default fallback: `readback → host multiply → upload` for each
    /// requested side. CUDA overrides with two 1D NVRTC kernels.
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

    /// Wave 2.1: device-resident backward for `rms_norm(x, weight, eps)`.
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
    /// Default fallback: `readback → host loop → upload`. CUDA overrides
    /// with three NVRTC kernels (per-row inv_rms scratch, then per-row
    /// grad_x with shared-mem `dot` reduce, then per-col grad_w reduce).
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

        // Per-row inv_rms.
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

    /// Wave 2.1: device-resident backward for `rope(x, cos, sin)`. The
    /// backward is identical to the forward with `sin` negated:
    ///   grad_x = rope_forward(upstream, cos, -sin)
    ///
    /// `cos`/`sin` stay host-side (mirrors `rope` forward — caches are
    /// per-seq and seldom benefit from being device-resident).
    ///
    /// Default fallback: `readback(upstream) → host neg(sin) →
    /// cpu_rope_forward → upload`. CUDA overrides with a dedicated NVRTC
    /// kernel that inlines the sign flip.
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

fn shape_size(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

/// CPU reference implementation of row-major matmul (2D + batched 3D).
/// Exposed so other backends can reuse it as a fallback.
pub fn cpu_matmul_forward(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>)> {
    use crate::AutogradError;
    match (a_shape.len(), b_shape.len()) {
        (2, 2) => {
            let out_shape = matmul_output_shape(a_shape, b_shape)?;
            let m = a_shape[0];
            let k = a_shape[1];
            let n = b_shape[1];
            let mut out = vec![0.0f32; m * n];
            sgemm_row_major(m, k, n, a, b, &mut out);
            Ok((out, out_shape))
        }
        (3, 3) => {
            let out_shape = matmul_output_shape(a_shape, b_shape)?;
            let batch = a_shape[0];
            let m = a_shape[1];
            let k = a_shape[2];
            let n = b_shape[2];
            let mut out = vec![0.0f32; batch * m * n];
            let a_batch_stride = m * k;
            let b_batch_stride = k * n;
            let out_batch_stride = m * n;
            for batch_index in 0..batch {
                let a_base = batch_index * a_batch_stride;
                let b_base = batch_index * b_batch_stride;
                let out_base = batch_index * out_batch_stride;
                sgemm_row_major(
                    m,
                    k,
                    n,
                    &a[a_base..a_base + a_batch_stride],
                    &b[b_base..b_base + b_batch_stride],
                    &mut out[out_base..out_base + out_batch_stride],
                );
            }
            Ok((out, out_shape))
        }
        _ => Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2 or rank-3",
            got: a_shape.len().max(b_shape.len()),
        }),
    }
}

pub fn matmul_bt_output_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    use crate::AutogradError;
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2",
            got: a_shape.len().max(b_shape.len()),
        });
    }
    let k_a = a_shape[1];
    let k_b = b_shape[1];
    if k_a != k_b {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![k_a],
            got: vec![k_b],
        });
    }
    Ok(vec![a_shape[0], b_shape[0]])
}

/// CPU `C = A @ B^T` for rank-2 row-major tensors without materialising `B^T`.
/// Shapes: `A:[M,K]`, `B:[N,K]`, output `[M,N]`.
pub fn cpu_matmul_bt_forward(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>)> {
    let out_shape = matmul_bt_output_shape(a_shape, b_shape)?;
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];
    let expected_a = m * k;
    let expected_b = n * k;
    if a.len() != expected_a {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: a.len(),
            shape: a_shape.to_vec(),
            size: expected_a,
        });
    }
    if b.len() != expected_b {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: b.len(),
            shape: b_shape.to_vec(),
            size: expected_b,
        });
    }
    let mut out = vec![0.0f32; m * n];
    matmul_a_bt_into(a, a_shape_2d(m, k), b, b_shape_2d(n, k), &mut out);
    Ok((out, out_shape))
}

/// Row-major `C = A @ B` for one rank-2 sgemm tile. OPD-shape-aware dispatch:
///   - **Saxpy inline loop** for thin matmuls (`n < SAXPY_N_THRESHOLD`) and
///     single-row matmuls (`m == 1`). Hits ~20 GFLOPs/s on Zen 2 for
///     cache-resident OPD projection shapes, and avoids matrixmultiply's
///     pack overhead in the M=1 rollout-last-row lm_head regime.
///   - **`matrixmultiply::sgemm`** for `n >= SAXPY_N_THRESHOLD`. With
///     `lm_head`'s `N=151936` the saxpy thrashes L1 (608 KB per B row);
///     matrixmultiply's tile-pack reuses A across N-tiles and pushes lm_head
///     forward from ~8 GFLOPs/s saxpy ceiling to ~16 GFLOPs/s on Zen 2.
///
/// Caller guarantees `a.len() == m*k`, `b.len() == k*n`, `out.len() == m*n` and
/// that `out` is already zero-initialised.
fn sgemm_row_major(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], out: &mut [f32]) {
    /// Crossover N where matrixmultiply's pack-A / pack-B is amortised by
    /// the extra cache reuse. Empirically at M=4 Qwen3-0.6B shapes on Zen 2:
    /// gate_proj (N=3072) wins with saxpy; lm_head (N=151936) wins with
    /// matrixmultiply. Loose bracket so future model shapes between 3K and
    /// 30K take the lower-overhead saxpy path.
    const SAXPY_N_THRESHOLD: usize = 32_768;
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    if m == 1 || n < SAXPY_N_THRESHOLD {
        for row in 0..m {
            let a_row = &a[row * k..(row + 1) * k];
            let out_row = &mut out[row * n..(row + 1) * n];
            for inner in 0..k {
                let a_value = a_row[inner];
                let b_row = &b[inner * n..(inner + 1) * n];
                for col in 0..n {
                    out_row[col] += a_value * b_row[col];
                }
            }
        }
        return;
    }
    // Safety: a/b are row-major contiguous slices of length m*k / k*n; `out`
    // is row-major contiguous m*n; beta=0 means the pre-existing `C` values
    // are unread.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// CPU reference matmul backward. Computes `grad_a = grad_out @ B^T` and
/// `grad_b = A^T @ grad_out`. Physically transposes the last two axes of the
/// saved operand on the host and then calls `cpu_matmul_forward` — this is
/// the authoritative numerical reference every GPU backend must match.
///
/// `need_grad_a`/`need_grad_b` skip the corresponding SGEMM when false; the
/// returned `Vec<f32>` is empty in that case so callers can cheaply detect
/// "no grad produced" without allocating.
pub fn cpu_matmul_backward(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    grad_out: &[f32],
    grad_out_shape: &[usize],
    need_grad_a: bool,
    need_grad_b: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    use crate::AutogradError;
    let expected_out = matmul_output_shape(a_shape, b_shape)?;
    if grad_out_shape != expected_out.as_slice() {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_out,
            got: grad_out_shape.to_vec(),
        });
    }

    match (a_shape.len(), b_shape.len()) {
        (2, 2) => {
            let m = a_shape[0];
            let k = a_shape[1];
            let n = b_shape[1];
            let grad_a = if need_grad_a {
                let mut out = vec![0.0f32; m * k];
                matmul_a_bt_into(grad_out, a_shape_2d(m, n), b, b_shape_2d(k, n), &mut out);
                out
            } else {
                Vec::new()
            };
            let grad_b = if need_grad_b {
                let mut out = vec![0.0f32; k * n];
                matmul_at_b_into(a, a_shape_2d(m, k), grad_out, b_shape_2d(m, n), &mut out);
                out
            } else {
                Vec::new()
            };
            Ok((grad_a, grad_b))
        }
        (3, 3) => {
            let batch = a_shape[0];
            let m = a_shape[1];
            let k = a_shape[2];
            let n = b_shape[2];
            let a_plane = m * k;
            let b_plane = k * n;
            let grad_out_plane = m * n;
            let grad_a = if need_grad_a {
                let mut out = vec![0.0f32; batch * m * k];
                for bi in 0..batch {
                    matmul_a_bt_into(
                        &grad_out[bi * grad_out_plane..(bi + 1) * grad_out_plane],
                        a_shape_2d(m, n),
                        &b[bi * b_plane..(bi + 1) * b_plane],
                        b_shape_2d(k, n),
                        &mut out[bi * a_plane..(bi + 1) * a_plane],
                    );
                }
                out
            } else {
                Vec::new()
            };
            let grad_b = if need_grad_b {
                let mut out = vec![0.0f32; batch * k * n];
                for bi in 0..batch {
                    matmul_at_b_into(
                        &a[bi * a_plane..(bi + 1) * a_plane],
                        a_shape_2d(m, k),
                        &grad_out[bi * grad_out_plane..(bi + 1) * grad_out_plane],
                        b_shape_2d(m, n),
                        &mut out[bi * b_plane..(bi + 1) * b_plane],
                    );
                }
                out
            } else {
                Vec::new()
            };
            Ok((grad_a, grad_b))
        }
        _ => Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2 or rank-3",
            got: a_shape.len().max(b_shape.len()),
        }),
    }
}

pub fn cpu_matmul_bt_backward(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    grad_out: &[f32],
    grad_out_shape: &[usize],
    need_grad_a: bool,
    need_grad_b: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    use crate::AutogradError;
    let expected_out = matmul_bt_output_shape(a_shape, b_shape)?;
    if grad_out_shape != expected_out.as_slice() {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_out,
            got: grad_out_shape.to_vec(),
        });
    }

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];
    let grad_a = if need_grad_a {
        let (grad_a, _) = cpu_matmul_forward(grad_out, &[m, n], b, &[n, k])?;
        grad_a
    } else {
        Vec::new()
    };
    let grad_b = if need_grad_b {
        let mut out = vec![0.0f32; n * k];
        matmul_at_b_into(grad_out, a_shape_2d(m, n), a, b_shape_2d(m, k), &mut out);
        out
    } else {
        Vec::new()
    };
    Ok((grad_a, grad_b))
}

/// Pair of rank-2 row dimensions, kept inline so the rank-3 dispatcher can
/// reuse the same kernel without re-allocating shape `Vec`s on every batch.
#[inline]
fn a_shape_2d(rows: usize, cols: usize) -> (usize, usize) {
    (rows, cols)
}
#[inline]
fn b_shape_2d(rows: usize, cols: usize) -> (usize, usize) {
    (rows, cols)
}

/// Compute `out = a @ b^T` for row-major rank-2 buffers **without materialising
/// `b^T`**. Dispatches to `matrixmultiply::sgemm` with a strided view of `b`:
/// passing `rsb = 1, csb = N_phys` re-interprets the row-major `[K_phys, N_phys]`
/// buffer as the transposed `[N_phys, K_phys]` matrix without copying. Used by
/// `cpu_matmul_backward` for `grad_a = grad_out @ B^T`.
///
/// Shapes (caller-enforced):
/// - `a`: `[M, N]` (row-major contiguous, len `M * N`)
/// - `b`: `[K, N]` (row-major contiguous, len `K * N`) — logical pre-transpose
/// - `out`: `[M, K]` (row-major contiguous, len `M * K`, pre-zeroed)
#[inline]
fn matmul_a_bt_into(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
    out: &mut [f32],
) {
    let (m, n_a) = a_shape;
    let (k, n_b) = b_shape;
    debug_assert_eq!(n_a, n_b, "a and b must share the K-equivalent dim");
    let n = n_a;
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    // Safety: a/b are row-major contiguous slices of length m*n / k*n; the
    // strided `b` view (rsb=1, csb=n) addresses every element once via
    // `b_ptr[k_log * 1 + n_log * n]`, which equals `b_ptr[n_log * n + k_log]`
    // = the (n_log, k_log) entry of the logical transpose. beta=0 means the
    // pre-existing `out` values are unread.
    unsafe {
        matrixmultiply::sgemm(
            m,
            n,
            k,
            1.0,
            a.as_ptr(),
            n as isize,
            1,
            b.as_ptr(),
            1,
            n as isize,
            0.0,
            out.as_mut_ptr(),
            k as isize,
            1,
        );
    }
}

/// Compute `out = a^T @ b` for row-major rank-2 buffers **without materialising
/// `a^T`**. Dispatches to `matrixmultiply::sgemm` with a strided view of `a`:
/// passing `rsa = 1, csa = K_phys` re-interprets the row-major `[M_phys, K_phys]`
/// buffer as the transposed `[K_phys, M_phys]` matrix without copying. Used by
/// `cpu_matmul_backward` for `grad_b = A^T @ grad_out`.
///
/// Shapes (caller-enforced):
/// - `a`: `[M, K]` (row-major contiguous, len `M * K`) — logical pre-transpose
/// - `b`: `[M, N]` (row-major contiguous, len `M * N`)
/// - `out`: `[K, N]` (row-major contiguous, len `K * N`, pre-zeroed)
#[inline]
fn matmul_at_b_into(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
    out: &mut [f32],
) {
    let (m_a, k) = a_shape;
    let (m_b, n) = b_shape;
    debug_assert_eq!(m_a, m_b, "a and b must share the M dim");
    let m = m_a;
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    // Safety: a/b are row-major contiguous slices of length m*k / m*n; the
    // strided `a` view (rsa=1, csa=k) addresses `a_ptr[k_log * 1 + m_log * k]`
    // = `a_ptr[m_log * k + k_log]` = the (m_log, k_log) entry of physical A,
    // which is the (k_log, m_log) entry of logical A^T. beta=0 means the
    // pre-existing `out` values are unread.
    unsafe {
        matrixmultiply::sgemm(
            k,
            m,
            n,
            1.0,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// CPU reference for row-wise softmax over the last axis. Matches the
/// numerically-stable implementation in `ops::softmax::softmax` so that
/// backends can fall back to this when GPU acceleration is unavailable.
pub fn cpu_softmax_forward_last_axis(x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let rows = x.len() / last_dim;
    let mut out = vec![0.0f32; x.len()];
    for row in 0..rows {
        let base = row * last_dim;
        let slice = &x[base..base + last_dim];
        let max_value = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = slice
            .iter()
            .map(|value| (*value - max_value).exp())
            .sum::<f32>();
        for col in 0..last_dim {
            out[base + col] = (slice[col] - max_value).exp() / denom;
        }
    }
    Ok(out)
}

/// CPU reference for row-wise log-softmax over the last axis.
pub fn cpu_log_softmax_forward_last_axis(x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let rows = x.len() / last_dim;
    let mut out = vec![0.0f32; x.len()];
    for row in 0..rows {
        let base = row * last_dim;
        let slice = &x[base..base + last_dim];
        let max_value = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = slice
            .iter()
            .map(|value| (*value - max_value).exp())
            .sum::<f32>();
        let log_denom = denom.ln();
        for col in 0..last_dim {
            out[base + col] = (slice[col] - max_value) - log_denom;
        }
    }
    Ok(out)
}

/// CPU reference for `log_softmax_last_axis_backward`. Computes
/// `grad_input[i, j] = upstream[i, j] - exp(log_softmax_output[i, j]) * sum_j(upstream[i, j])`
/// row-wise over the last axis. `log_softmax_output` is the saved
/// forward output — `softmax(x) = exp(log_softmax(x))`, so the
/// derivative identity reuses it without recomputing softmax.
///
/// Mirrors the inline math in `ops::softmax::log_softmax_backward`
/// (host-eager path). Kept as a free function so the device-handle
/// fallback in `Backend::log_softmax_last_axis_backward` can reuse the
/// same reference and parity tests can compare device against CPU.
pub fn cpu_log_softmax_backward(
    upstream: &[f32],
    log_softmax_output: &[f32],
    shape: &[usize],
) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected = shape_size(shape);
    if upstream.len() != expected {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: upstream.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }
    if log_softmax_output.len() != expected {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: log_softmax_output.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }
    let rows = expected / last_dim;
    let mut grad = vec![0.0_f32; expected];
    for row in 0..rows {
        let base = row * last_dim;
        let mut sum_grad = 0.0_f32;
        for col in 0..last_dim {
            sum_grad += upstream[base + col];
        }
        for col in 0..last_dim {
            grad[base + col] =
                upstream[base + col] - log_softmax_output[base + col].exp() * sum_grad;
        }
    }
    Ok(grad)
}

/// CPU reference for `softmax_last_axis_backward`. Computes
/// `grad_input[i, j] = y[i, j] * (upstream[i, j] - sum_j(upstream[i, j] * y[i, j]))`
/// row-wise over the last axis. `softmax_output` is the saved forward output.
pub fn cpu_softmax_backward(
    upstream: &[f32],
    softmax_output: &[f32],
    shape: &[usize],
) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected = shape_size(shape);
    if upstream.len() != expected {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: upstream.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }
    if softmax_output.len() != expected {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: softmax_output.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }
    let rows = expected / last_dim;
    let mut grad = vec![0.0_f32; expected];
    for row in 0..rows {
        let base = row * last_dim;
        let mut dot = 0.0_f32;
        for col in 0..last_dim {
            dot += upstream[base + col] * softmax_output[base + col];
        }
        for col in 0..last_dim {
            grad[base + col] = softmax_output[base + col] * (upstream[base + col] - dot);
        }
    }
    Ok(grad)
}

/// CPU reference `out = a * b` for equal-length contiguous slices.
pub fn cpu_mul_forward(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![a.len()],
            got: vec![b.len()],
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
}

/// CPU reference `out = a * s`.
pub fn cpu_mul_scalar_forward(a: &[f32], s: f32) -> Result<Vec<f32>> {
    Ok(a.iter().map(|x| x * s).collect())
}

/// CPU reference right-aligned broadcast-add.
///
/// Output shape equals `a_shape`; `b` is broadcast into `a`. `b_shape.len()`
/// must be `<= a_shape.len()`; each matching `b`-axis must be either `1` or
/// equal to the corresponding `a`-axis. See `broadcast_offset` for the
/// index rule.
pub fn cpu_add_broadcast_forward(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<Vec<f32>> {
    validate_broadcast(a_shape, b_shape)?;
    let a_size: usize = shape_size(a_shape);
    let b_size: usize = shape_size(b_shape);
    if a.len() != a_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: a.len(),
            shape: a_shape.to_vec(),
            size: a_size,
        });
    }
    if b.len() != b_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: b.len(),
            shape: b_shape.to_vec(),
            size: b_size,
        });
    }
    let mut out = vec![0.0f32; a_size];
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = a[index] + b[broadcast_offset(index, a_shape, b_shape)];
    }
    Ok(out)
}

/// Validate that `b_shape` is right-aligned broadcast-compatible into `a_shape`.
pub(crate) fn validate_broadcast(a_shape: &[usize], b_shape: &[usize]) -> Result<()> {
    if b_shape.len() > a_shape.len() {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: a_shape.to_vec(),
            got: b_shape.to_vec(),
        });
    }

    let rank_offset = a_shape.len() - b_shape.len();
    for (index, &dim) in b_shape.iter().enumerate() {
        let target = a_shape[rank_offset + index];
        if dim != 1 && dim != target {
            return Err(crate::AutogradError::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            });
        }
    }

    Ok(())
}

/// Map an output linear index in `out_shape` to the corresponding flat offset
/// into a right-aligned broadcast operand with shape `b_shape`.
pub(crate) fn broadcast_offset(out_index: usize, out_shape: &[usize], b_shape: &[usize]) -> usize {
    if b_shape.is_empty() {
        return 0;
    }

    let coords = linear_to_coords(out_index, out_shape);
    let rank_offset = out_shape.len() - b_shape.len();
    let b_strides = broadcast_strides(b_shape);
    let mut offset = 0usize;
    for (index, stride) in b_strides.iter().enumerate() {
        let coord = if b_shape[index] == 1 {
            0
        } else {
            coords[rank_offset + index]
        };
        offset += coord * stride;
    }
    offset
}

/// Row-major contiguous strides for `shape`. Shared helper used by broadcast
/// math (not the `Tensor` layout stride — that lives in `tensor.rs`).
pub(crate) fn broadcast_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for (index, dim) in shape.iter().enumerate().rev() {
        strides[index] = stride;
        stride *= *dim;
    }
    strides
}

/// Unravel a linear index into per-axis coordinates (row-major).
pub(crate) fn linear_to_coords(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut coords = vec![0; shape.len()];
    for index in (0..shape.len()).rev() {
        let dim = shape[index];
        coords[index] = linear % dim;
        linear /= dim;
    }
    coords
}

/// CPU reference `out = exp(a)`.
pub fn cpu_exp_forward(a: &[f32]) -> Result<Vec<f32>> {
    Ok(a.iter().map(|x| x.exp()).collect())
}

/// CPU reference `out = -a`.
pub fn cpu_neg_forward(a: &[f32]) -> Result<Vec<f32>> {
    Ok(a.iter().map(|x| -x).collect())
}

/// CPU reference GELU (tanh approximation). Matches the CUDA `gelu_f32` kernel.
pub fn cpu_gelu_forward(a: &[f32]) -> Result<Vec<f32>> {
    const K: f32 = 0.797_884_6_f32; // sqrt(2/pi)
    Ok(a.iter()
        .map(|&x| {
            let inner = K * (x + 0.044_715_f32 * x * x * x);
            0.5_f32 * x * (1.0_f32 + inner.tanh())
        })
        .collect())
}

/// CPU reference SiLU (Swish): `out = a * sigmoid(a)`.
pub fn cpu_silu_forward(a: &[f32]) -> Result<Vec<f32>> {
    Ok(a.iter()
        .map(|&x| x * (1.0_f32 / (1.0_f32 + (-x).exp())))
        .collect())
}

/// CPU reference sigmoid: `out = 1 / (1 + exp(-a))`.
pub fn cpu_sigmoid_forward(a: &[f32]) -> Result<Vec<f32>> {
    Ok(a.iter()
        .map(|&x| 1.0_f32 / (1.0_f32 + (-x).exp()))
        .collect())
}

/// CPU reference RMSNorm over the last axis.
pub fn cpu_rms_norm_forward(
    x: &[f32],
    weight: &[f32],
    shape: &[usize],
    eps: f32,
) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected: usize = shape.iter().product();
    if x.len() != expected {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![x.len()],
        });
    }
    if weight.len() != last_dim {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![last_dim],
            got: vec![weight.len()],
        });
    }

    let rows = expected / last_dim;
    let mut out = vec![0.0_f32; expected];
    for row in 0..rows {
        let base = row * last_dim;
        let slice = &x[base..base + last_dim];
        let mean_sq = slice.iter().map(|v| v * v).sum::<f32>() / last_dim as f32;
        let inv_rms = (mean_sq + eps).sqrt().recip();
        for col in 0..last_dim {
            out[base + col] = slice[col] * inv_rms * weight[col];
        }
    }
    Ok(out)
}

/// CPU reference embedding gather. Returns `[n_ids * dim]` row-major; ids out
/// of range produce a zero row (matches the CUDA kernel's behavior).
pub fn cpu_embedding_forward(
    weight: &[f32],
    vocab: usize,
    dim: usize,
    ids: &[i32],
) -> Result<Vec<f32>> {
    if weight.len() != vocab * dim {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![vocab * dim],
            got: vec![weight.len()],
        });
    }
    let mut out = vec![0.0_f32; ids.len() * dim];
    for (row, &id) in ids.iter().enumerate() {
        if id < 0 {
            continue;
        }
        let id = id as usize;
        if id >= vocab {
            continue;
        }
        let src = &weight[id * dim..(id + 1) * dim];
        let dst = &mut out[row * dim..(row + 1) * dim];
        dst.copy_from_slice(src);
    }
    Ok(out)
}

/// CPU reference sum over the last axis.
pub fn cpu_sum_last_axis_forward(x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected: usize = shape.iter().product();
    if x.len() != expected {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![x.len()],
        });
    }
    let rows = expected / last_dim;
    let mut out = vec![0.0_f32; rows];
    for (row, slot) in out.iter_mut().enumerate().take(rows) {
        let base = row * last_dim;
        *slot = x[base..base + last_dim].iter().sum();
    }
    Ok(out)
}

/// CPU reference mean over the last axis.
pub fn cpu_mean_last_axis_forward(x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    let mut out = cpu_sum_last_axis_forward(x, shape)?;
    let inv = 1.0_f32 / last_dim as f32;
    for v in out.iter_mut() {
        *v *= inv;
    }
    Ok(out)
}

/// CPU reference for NeoX RoPE (matches `ops::rope::rope` — element `i` pairs
/// with `i + half_dim`). `x_shape = [batch, heads, seq, head_dim]`; `cos`/`sin`
/// are `[seq, half_dim]` row-major.
pub fn cpu_rope_forward(
    x: &[f32],
    x_shape: &[usize],
    cos: &[f32],
    sin: &[f32],
) -> Result<Vec<f32>> {
    use crate::AutogradError;
    if x_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: x_shape.len(),
        });
    }
    let batch = x_shape[0];
    let heads = x_shape[1];
    let seq = x_shape[2];
    let head_dim = x_shape[3];
    if !head_dim.is_multiple_of(2) {
        return Err(AutogradError::InvalidRank {
            expected: "even head dim",
            got: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    let expected_x = batch * heads * seq * head_dim;
    if x.len() != expected_x {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected_x],
            got: vec![x.len()],
        });
    }
    if cos.len() != sin.len() || !cos.len().is_multiple_of(seq.max(1)) {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![seq * half_dim],
            got: vec![cos.len().min(sin.len())],
        });
    }
    let rotary_half_dim = cos.len() / seq.max(1);
    if rotary_half_dim == 0 || rotary_half_dim > half_dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![seq * half_dim],
            got: vec![cos.len()],
        });
    }
    let rotary_dim = rotary_half_dim * 2;
    let mut out = vec![0.0_f32; expected_x];
    for b in 0..batch {
        for h in 0..heads {
            for t in 0..seq {
                let rope_base = t * rotary_half_dim;
                let base = (((b * heads) + h) * seq + t) * head_dim;
                for i in 0..rotary_half_dim {
                    let x0 = x[base + i];
                    let x1 = x[base + i + rotary_half_dim];
                    let c = cos[rope_base + i];
                    let s = sin[rope_base + i];
                    out[base + i] = (x0 * c) - (x1 * s);
                    out[base + i + rotary_half_dim] = (x1 * c) + (x0 * s);
                }
                out[(base + rotary_dim)..(base + head_dim)]
                    .copy_from_slice(&x[(base + rotary_dim)..(base + head_dim)]);
            }
        }
    }
    Ok(out)
}

/// CPU reference gather along the last axis.
/// `out[prefix] = src[prefix * vocab + ids[prefix]]`. Out-of-range or negative
/// ids produce an error (unlike embedding which zero-fills — the caller is
/// responsible for validating ids).
pub fn cpu_gather_last_dim_forward(
    src: &[f32],
    src_shape: &[usize],
    ids: &[i32],
) -> Result<Vec<f32>> {
    use crate::AutogradError;
    if src_shape.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        });
    }
    let vocab = *src_shape.last().expect("non-empty shape above");
    let prefix: usize = src_shape[..src_shape.len() - 1]
        .iter()
        .product::<usize>()
        .max(1);
    let expected: usize = src_shape.iter().product();
    if src.len() != expected {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![src.len()],
        });
    }
    if ids.len() != prefix {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix,
            got: ids.len(),
        });
    }
    let mut out = vec![0.0_f32; prefix];
    for (i, &id) in ids.iter().enumerate() {
        if id < 0 || (id as usize) >= vocab {
            return Err(AutogradError::IndexOutOfBounds {
                index: id as usize,
                upper: vocab,
            });
        }
        out[i] = src[i * vocab + id as usize];
    }
    Ok(out)
}

/// CPU reference for `gather_last_dim_backward`. Zero-fills a
/// `src_shape = [prefix..., vocab]` buffer then writes
/// `upstream[row]` into `out[row * vocab + indices[row]]` for each
/// prefix position. Equivalent to the `scatter_add_rows_forward` call
/// in `ops::gather::gather_last_dim_backward` with `feature_dim = 1`
/// and remapped flat ids — kept as a dedicated function so the device
/// backward override returns the same `[B, S, V]` grad shape the
/// autograd graph expects without needing the caller to know about
/// the flat-id trick.
///
/// Negative or out-of-range indices are silently skipped (matches
/// `cpu_scatter_add_rows_forward` and the CUDA kernel's OOB handling).
pub fn cpu_gather_last_dim_backward(
    upstream: &[f32],
    indices: &[i32],
    src_shape: &[usize],
) -> Result<Vec<f32>> {
    use crate::AutogradError;
    if src_shape.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        });
    }
    let vocab = *src_shape.last().expect("non-empty shape above");
    let prefix: usize = src_shape[..src_shape.len() - 1]
        .iter()
        .product::<usize>()
        .max(1);
    if upstream.len() != prefix {
        return Err(AutogradError::DataLengthMismatch {
            len: upstream.len(),
            shape: src_shape[..src_shape.len() - 1].to_vec(),
            size: prefix,
        });
    }
    if indices.len() != prefix {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix,
            got: indices.len(),
        });
    }
    let total = prefix * vocab;
    let mut grad = vec![0.0_f32; total];
    for (row, &id) in indices.iter().enumerate() {
        if id < 0 {
            continue;
        }
        let id_usize = id as usize;
        if id_usize >= vocab {
            continue;
        }
        grad[row * vocab + id_usize] = upstream[row];
    }
    Ok(grad)
}

/// CPU reference scatter-add into a `[vocab, feature_dim]` output.
///
/// `upstream` has length `prefix_rows * feature_dim`; `indices.len() == prefix_rows`.
/// For each row, the feature slice is added into the bin selected by the
/// corresponding index. Negative or out-of-range indices are silently
/// skipped — matches the prior inline scatter in `embedding_backward`
/// (which bounds-checked at the op layer) and the CUDA kernel's OOB
/// handling so behavior is identical across backends.
pub fn cpu_scatter_add_rows_forward(
    upstream: &[f32],
    prefix_rows: usize,
    feature_dim: usize,
    indices: &[i32],
    vocab: usize,
) -> Result<Vec<f32>> {
    let expected_upstream = prefix_rows * feature_dim;
    if upstream.len() != expected_upstream {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![expected_upstream],
            got: vec![upstream.len()],
        });
    }
    if indices.len() != prefix_rows {
        return Err(crate::AutogradError::InvalidIndicesLen {
            expected: prefix_rows,
            got: indices.len(),
        });
    }
    let mut out = vec![0.0_f32; vocab * feature_dim];
    for (row, &id) in indices.iter().enumerate() {
        if id < 0 {
            continue;
        }
        let id = id as usize;
        if id >= vocab {
            continue;
        }
        let src_base = row * feature_dim;
        let dst_base = id * feature_dim;
        for col in 0..feature_dim {
            out[dst_base + col] += upstream[src_base + col];
        }
    }
    Ok(out)
}

/// CPU reference transpose-swap: swap `axis1` and `axis2` of a contiguous
/// row-major tensor with shape `old_shape`. Returns `(data, new_shape)`.
/// Used by the `Backend::transpose_axes_swap` default fallback and by the
/// ops-layer host-eager path — keeping both on the same function means the
/// device-default-fallback and the host path produce byte-identical output
/// for a given input.
pub fn cpu_transpose_swap(
    data: &[f32],
    old_shape: &[usize],
    axis1: usize,
    axis2: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let rank = old_shape.len();
    if axis1 >= rank {
        return Err(crate::AutogradError::AxisOutOfBounds { axis: axis1, rank });
    }
    if axis2 >= rank {
        return Err(crate::AutogradError::AxisOutOfBounds { axis: axis2, rank });
    }
    if axis1 == axis2 {
        return Ok((data.to_vec(), old_shape.to_vec()));
    }

    let mut new_shape = old_shape.to_vec();
    new_shape.swap(axis1, axis2);

    // Contiguous strides over `old_shape` — the source we're reading from.
    let mut old_strides = vec![0usize; rank];
    let mut stride = 1usize;
    for (index, dim) in old_shape.iter().enumerate().rev() {
        old_strides[index] = stride;
        stride *= *dim;
    }

    let mut out = vec![0.0_f32; data.len()];
    for (out_index, slot) in out.iter_mut().enumerate() {
        // Decompose out_index into new_shape coords, then swap the two
        // axes to recover the original source coords.
        let mut coords = vec![0usize; rank];
        let mut linear = out_index;
        for axis in (0..rank).rev() {
            let dim = new_shape[axis];
            coords[axis] = linear % dim;
            linear /= dim;
        }
        coords.swap(axis1, axis2);
        let input_index: usize = coords
            .iter()
            .zip(old_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        *slot = data[input_index];
    }
    Ok((out, new_shape))
}

/// CPU reference contiguous slice: copy elements of `data` (row-major over
/// `old_shape`) whose per-axis coordinate is in `[starts[i], ends[i])`.
/// Returns `(sliced_data, new_shape)` with `new_shape[i] = ends[i] - starts[i]`.
/// Used by the `Backend::slice` default fallback so device-default and host
/// paths share one numerical reference. M5.3b.16.
pub fn cpu_slice(
    data: &[f32],
    old_shape: &[usize],
    starts: &[usize],
    ends: &[usize],
) -> Result<(Vec<f32>, Vec<usize>)> {
    let rank = old_shape.len();
    let new_shape = validate_slice_shape(old_shape, starts, ends)?;
    let new_numel: usize = if new_shape.is_empty() {
        1
    } else {
        new_shape.iter().product()
    };

    let mut old_strides = vec![0usize; rank];
    let mut stride = 1usize;
    for (index, dim) in old_shape.iter().enumerate().rev() {
        old_strides[index] = stride;
        stride *= *dim;
    }

    let mut out = vec![0.0_f32; new_numel];
    for (out_index, slot) in out.iter_mut().enumerate() {
        let mut coords = vec![0usize; rank];
        let mut linear = out_index;
        for axis in (0..rank).rev() {
            let dim = new_shape[axis];
            if dim > 0 {
                coords[axis] = linear % dim;
                linear /= dim;
            }
        }
        let input_index: usize = coords
            .iter()
            .enumerate()
            .map(|(axis, &c)| (c + starts[axis]) * old_strides[axis])
            .sum();
        *slot = data[input_index];
    }
    Ok((out, new_shape))
}

/// CPU reference for KV-cache append: concatenate two rank-4 contiguous
/// tensors shaped `[batch, heads, seq, dim]` along axis 2.
pub fn cpu_concat_axis2(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>)> {
    if a_shape.len() != 4 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "4",
            got: a_shape.len(),
        });
    }
    if b_shape.len() != 4 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "4",
            got: b_shape.len(),
        });
    }
    if a_shape[0] != b_shape[0] || a_shape[1] != b_shape[1] || a_shape[3] != b_shape[3] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![a_shape[0], a_shape[1], a_shape[3]],
            got: vec![b_shape[0], b_shape[1], b_shape[3]],
        });
    }
    let a_size = shape_size(a_shape);
    if a.len() != a_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: a.len(),
            shape: a_shape.to_vec(),
            size: a_size,
        });
    }
    let b_size = shape_size(b_shape);
    if b.len() != b_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: b.len(),
            shape: b_shape.to_vec(),
            size: b_size,
        });
    }

    let batch = a_shape[0];
    let heads = a_shape[1];
    let a_seq = a_shape[2];
    let b_seq = b_shape[2];
    let dim = a_shape[3];
    let out_shape = vec![batch, heads, a_seq + b_seq, dim];
    let mut out = vec![0.0_f32; shape_size(&out_shape)];

    for batch_idx in 0..batch {
        for head_idx in 0..heads {
            let out_base = ((batch_idx * heads + head_idx) * (a_seq + b_seq)) * dim;
            let a_base = ((batch_idx * heads + head_idx) * a_seq) * dim;
            let b_base = ((batch_idx * heads + head_idx) * b_seq) * dim;
            let a_len = a_seq * dim;
            let b_len = b_seq * dim;
            out[out_base..out_base + a_len].copy_from_slice(&a[a_base..a_base + a_len]);
            out[out_base + a_len..out_base + a_len + b_len]
                .copy_from_slice(&b[b_base..b_base + b_len]);
        }
    }

    Ok((out, out_shape))
}

/// CPU reference for decode-time GQA causal attention.
#[allow(clippy::too_many_arguments)]
pub fn cpu_causal_sdpa_decode_gqa(
    q: &[f32],
    q_shape: &[usize],
    k: &[f32],
    k_shape: &[usize],
    v: &[f32],
    v_shape: &[usize],
    q_start: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    validate_decode_gqa_shapes(q_shape, k_shape, v_shape, q_start)?;
    let q_size = shape_size(q_shape);
    let k_size = shape_size(k_shape);
    let v_size = shape_size(v_shape);
    if q.len() != q_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: q.len(),
            shape: q_shape.to_vec(),
            size: q_size,
        });
    }
    if k.len() != k_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: k.len(),
            shape: k_shape.to_vec(),
            size: k_size,
        });
    }
    if v.len() != v_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: v.len(),
            shape: v_shape.to_vec(),
            size: v_size,
        });
    }

    let batch = q_shape[0];
    let query_heads = q_shape[1];
    let kv_heads = k_shape[1];
    let kv_len = k_shape[2];
    let head_dim = q_shape[3];
    let kv_repeat = query_heads / kv_heads;
    let visible = (q_start + 1).min(kv_len);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let out_shape = vec![batch, query_heads, 1, head_dim];
    let mut out = vec![0.0_f32; shape_size(&out_shape)];
    let mut scores = vec![0.0_f32; visible];

    for batch_idx in 0..batch {
        for query_head in 0..query_heads {
            let kv_head = query_head / kv_repeat;
            let q_base = (batch_idx * query_heads + query_head) * head_dim;

            let mut max_score = f32::NEG_INFINITY;
            for (pos, score_slot) in scores.iter_mut().enumerate().take(visible) {
                let k_base = ((batch_idx * kv_heads + kv_head) * kv_len + pos) * head_dim;
                let mut dot = 0.0_f32;
                for dim in 0..head_dim {
                    dot += q[q_base + dim] * k[k_base + dim];
                }
                let score = dot * scale;
                *score_slot = score;
                max_score = max_score.max(score);
            }

            let mut denom = 0.0_f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denom += *score;
            }
            let out_base = (batch_idx * query_heads + query_head) * head_dim;
            if denom == 0.0 {
                continue;
            }

            for dim in 0..head_dim {
                let mut acc = 0.0_f32;
                for (pos, &weight_exp) in scores.iter().enumerate() {
                    let v_base = ((batch_idx * kv_heads + kv_head) * kv_len + pos) * head_dim;
                    acc += (weight_exp / denom) * v[v_base + dim];
                }
                out[out_base + dim] = acc;
            }
        }
    }

    Ok((out, out_shape))
}

#[allow(clippy::too_many_arguments)]
fn cpu_qwen_decode_prepare_q(
    q_full: &[f32],
    q_full_shape: &[usize],
    q_norm_weight: &[f32],
    q_norm_weight_shape: &[usize],
    cos: &[f32],
    cos_shape: &[usize],
    sin: &[f32],
    sin_shape: &[usize],
    query_heads: usize,
    head_dim: usize,
    gated: bool,
    eps: f32,
) -> Result<QwenDecodePrepareQHost> {
    validate_qwen_decode_prepare_q_shapes(
        q_full_shape,
        q_norm_weight_shape,
        cos_shape,
        sin_shape,
        query_heads,
        head_dim,
        gated,
    )?;
    let q_full_size = shape_size(q_full_shape);
    if q_full.len() != q_full_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: q_full.len(),
            shape: q_full_shape.to_vec(),
            size: q_full_size,
        });
    }
    if q_norm_weight.len() != head_dim {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: q_norm_weight.len(),
            shape: q_norm_weight_shape.to_vec(),
            size: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    if cos.len() != half_dim || sin.len() != half_dim {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: cos.len().max(sin.len()),
            shape: vec![1, half_dim],
            size: half_dim,
        });
    }

    let batch = q_full_shape[0];
    let q_full_stride = q_full_shape[2];
    let head_stride = if gated { head_dim * 2 } else { head_dim };
    let out_shape = vec![batch, query_heads, 1, head_dim];
    let mut q_layout = vec![0.0_f32; shape_size(&out_shape)];
    let mut gate_layout = gated.then(|| vec![0.0_f32; shape_size(&out_shape)]);

    for batch_idx in 0..batch {
        for head in 0..query_heads {
            let src_base = batch_idx * q_full_stride + head * head_stride;
            let out_base = (batch_idx * query_heads + head) * head_dim;
            q_layout[out_base..out_base + head_dim]
                .copy_from_slice(&q_full[src_base..src_base + head_dim]);
            if let Some(gate) = gate_layout.as_mut() {
                gate[out_base..out_base + head_dim]
                    .copy_from_slice(&q_full[src_base + head_dim..src_base + head_stride]);
            }
        }
    }

    let q_norm_weight: Vec<f32> = q_norm_weight.iter().map(|&value| value + 1.0).collect();
    let q_normed = cpu_rms_norm_forward(&q_layout, &q_norm_weight, &out_shape, eps)?;
    let q_roped = cpu_rope_forward(&q_normed, &out_shape, cos, sin)?;
    Ok((q_roped, gate_layout, out_shape))
}

#[allow(clippy::too_many_arguments)]
fn cpu_qwen_decode_prepare_kv(
    k_full: &[f32],
    k_full_shape: &[usize],
    v_full: &[f32],
    v_full_shape: &[usize],
    k_norm_weight: &[f32],
    k_norm_weight_shape: &[usize],
    cos: &[f32],
    cos_shape: &[usize],
    sin: &[f32],
    sin_shape: &[usize],
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<QwenDecodePrepareKvHost> {
    validate_qwen_decode_prepare_kv_shapes(
        k_full_shape,
        v_full_shape,
        k_norm_weight_shape,
        cos_shape,
        sin_shape,
        kv_heads,
        head_dim,
    )?;
    let k_full_size = shape_size(k_full_shape);
    let v_full_size = shape_size(v_full_shape);
    if k_full.len() != k_full_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: k_full.len(),
            shape: k_full_shape.to_vec(),
            size: k_full_size,
        });
    }
    if v_full.len() != v_full_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: v_full.len(),
            shape: v_full_shape.to_vec(),
            size: v_full_size,
        });
    }
    if k_norm_weight.len() != head_dim {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: k_norm_weight.len(),
            shape: k_norm_weight_shape.to_vec(),
            size: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    if cos.len() != half_dim || sin.len() != half_dim {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: cos.len().max(sin.len()),
            shape: vec![1, half_dim],
            size: half_dim,
        });
    }

    let batch = k_full_shape[0];
    let full_stride = k_full_shape[2];
    let out_shape = vec![batch, kv_heads, 1, head_dim];
    let mut k_layout = vec![0.0_f32; shape_size(&out_shape)];
    let mut v_layout = vec![0.0_f32; shape_size(&out_shape)];

    for batch_idx in 0..batch {
        for head in 0..kv_heads {
            let src_base = batch_idx * full_stride + head * head_dim;
            let out_base = (batch_idx * kv_heads + head) * head_dim;
            k_layout[out_base..out_base + head_dim]
                .copy_from_slice(&k_full[src_base..src_base + head_dim]);
            v_layout[out_base..out_base + head_dim]
                .copy_from_slice(&v_full[src_base..src_base + head_dim]);
        }
    }

    let k_norm_weight: Vec<f32> = k_norm_weight.iter().map(|&value| value + 1.0).collect();
    let k_normed = cpu_rms_norm_forward(&k_layout, &k_norm_weight, &out_shape, eps)?;
    let k_roped = cpu_rope_forward(&k_normed, &out_shape, cos, sin)?;
    Ok((k_roped, v_layout, out_shape))
}

fn validate_qwen_decode_prepare_common(
    full_shape: &[usize],
    weight_shape: &[usize],
    cos_shape: &[usize],
    sin_shape: &[usize],
    heads: usize,
    head_dim: usize,
    projected_dim: usize,
) -> Result<()> {
    if full_shape.len() != 3 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "3",
            got: full_shape.len(),
        });
    }
    if full_shape[1] != 1 || full_shape[2] != projected_dim {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![full_shape[0], 1, projected_dim],
            got: full_shape.to_vec(),
        });
    }
    if heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(crate::AutogradError::TapeInvariant(
            "qwen decode prepare requires non-zero heads and even head_dim",
        ));
    }
    if weight_shape != [head_dim] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![head_dim],
            got: weight_shape.to_vec(),
        });
    }
    let rope_shape = [1, head_dim / 2];
    if cos_shape != rope_shape || sin_shape != rope_shape {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: rope_shape.to_vec(),
            got: cos_shape.to_vec(),
        });
    }
    Ok(())
}

pub(crate) fn validate_qwen_decode_prepare_q_shapes(
    q_full_shape: &[usize],
    q_norm_weight_shape: &[usize],
    cos_shape: &[usize],
    sin_shape: &[usize],
    query_heads: usize,
    head_dim: usize,
    gated: bool,
) -> Result<()> {
    let factor = if gated { 2 } else { 1 };
    validate_qwen_decode_prepare_common(
        q_full_shape,
        q_norm_weight_shape,
        cos_shape,
        sin_shape,
        query_heads,
        head_dim,
        query_heads * head_dim * factor,
    )
}

pub(crate) fn validate_qwen_decode_prepare_kv_shapes(
    k_full_shape: &[usize],
    v_full_shape: &[usize],
    k_norm_weight_shape: &[usize],
    cos_shape: &[usize],
    sin_shape: &[usize],
    kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    validate_qwen_decode_prepare_common(
        k_full_shape,
        k_norm_weight_shape,
        cos_shape,
        sin_shape,
        kv_heads,
        head_dim,
        kv_heads * head_dim,
    )?;
    if v_full_shape != k_full_shape {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: k_full_shape.to_vec(),
            got: v_full_shape.to_vec(),
        });
    }
    Ok(())
}

pub fn validate_decode_gqa_shapes(
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
    q_start: usize,
) -> Result<()> {
    for shape in [q_shape, k_shape, v_shape] {
        if shape.len() != 4 {
            return Err(crate::AutogradError::InvalidRank {
                expected: "4",
                got: shape.len(),
            });
        }
    }

    if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if q_shape[2] != 1 {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![1],
            got: vec![q_shape[2]],
        });
    }
    if q_shape[3] != k_shape[3] || q_shape[3] != v_shape[3] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if k_shape[1] != v_shape[1] || k_shape[2] != v_shape[2] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: k_shape.to_vec(),
            got: v_shape.to_vec(),
        });
    }
    if k_shape[2] == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-empty kv_len",
            got: 0,
        });
    }
    if q_shape[1] == 0 || k_shape[1] == 0 || !q_shape[1].is_multiple_of(k_shape[1]) {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![q_shape[1]],
            got: vec![k_shape[1]],
        });
    }
    if q_start >= k_shape[2] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![q_start + 1],
            got: vec![k_shape[2]],
        });
    }

    Ok(())
}

fn validate_slice_shape(
    old_shape: &[usize],
    starts: &[usize],
    ends: &[usize],
) -> Result<Vec<usize>> {
    let rank = old_shape.len();
    if starts.len() != rank {
        return Err(crate::AutogradError::InvalidIndicesLen {
            expected: rank,
            got: starts.len(),
        });
    }
    if ends.len() != rank {
        return Err(crate::AutogradError::InvalidIndicesLen {
            expected: rank,
            got: ends.len(),
        });
    }
    for ((&start, &end), &dim) in starts.iter().zip(ends.iter()).zip(old_shape.iter()) {
        if start > end {
            return Err(crate::AutogradError::TapeInvariant(
                "cpu_slice: start must be <= end for every axis",
            ));
        }
        if end > dim {
            return Err(crate::AutogradError::IndexOutOfBounds {
                index: end,
                upper: dim,
            });
        }
        if start > dim {
            return Err(crate::AutogradError::IndexOutOfBounds {
                index: start,
                upper: dim,
            });
        }
    }
    Ok(starts
        .iter()
        .zip(ends.iter())
        .map(|(&start, &end)| end - start)
        .collect())
}

/// CPU reference in-place AdamW update. Matches the formula in
/// `optim.rs::AdamW::step`:
///
/// - weight decay (decoupled): `param *= 1 - lr * wd`
/// - EMAs: `m = β1·m + (1-β1)·g`, `v = β2·v + (1-β2)·g²`
/// - bias-corrected step: `param -= lr · (m/bc1) / (√(v/bc2) + eps)`
///
/// Exposed as a free fn so `Backend::adamw_step` default impl and the
/// optimizer's host path share one numerical reference. Any backend
/// override (e.g. `MetalBackend::adamw_step`) MUST match this to the
/// 1e-5 gate enforced by `metal_adamw_step_stays_device_resident`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_adamw_step_in_place(
    param: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    grad: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    bc1: f32,
    bc2: f32,
) {
    debug_assert_eq!(param.len(), m.len());
    debug_assert_eq!(param.len(), v.len());
    debug_assert_eq!(param.len(), grad.len());

    if wd > 0.0 {
        let decay = 1.0 - (lr * wd);
        for value in param.iter_mut() {
            *value *= decay;
        }
    }

    for index in 0..param.len() {
        let g = grad[index];
        m[index] = (beta1 * m[index]) + ((1.0 - beta1) * g);
        v[index] = (beta2 * v[index]) + ((1.0 - beta2) * g * g);
        let m_hat = m[index] / bc1;
        let v_hat = v[index] / bc2;
        param[index] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

pub(crate) fn matmul_output_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    use crate::AutogradError;

    match (a_shape.len(), b_shape.len()) {
        (2, 2) => {
            if a_shape[1] != b_shape[0] {
                return Err(AutogradError::ShapeMismatch {
                    expected: vec![a_shape[1]],
                    got: vec![b_shape[0]],
                });
            }
            Ok(vec![a_shape[0], b_shape[1]])
        }
        (3, 3) => {
            if a_shape[0] != b_shape[0] {
                return Err(AutogradError::ShapeMismatch {
                    expected: vec![a_shape[0]],
                    got: vec![b_shape[0]],
                });
            }
            if a_shape[2] != b_shape[1] {
                return Err(AutogradError::ShapeMismatch {
                    expected: vec![a_shape[2]],
                    got: vec![b_shape[1]],
                });
            }
            Ok(vec![a_shape[0], a_shape[1], b_shape[2]])
        }
        _ => Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2 or rank-3",
            got: a_shape.len().max(b_shape.len()),
        }),
    }
}
