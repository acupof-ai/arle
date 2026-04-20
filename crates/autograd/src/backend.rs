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

        let _guard = crate::backend_metal::MLX_GUARD
            .lock()
            .expect("mlx guard poisoned");

        // Safety: `ptr` is owned by this handle, came from MLX FFI allocation,
        // and this Drop impl is the unique free path for the wrapped array.
        // `MLX_GUARD` serializes the free against all other MLX FFI calls.
        unsafe {
            mlx_sys::mlx_array_free(self.ptr);
        }
    }
}

#[cfg(feature = "metal")]
// Safety: `MlxHandle` owns an MLX array pointer. MLX's global stream is not
// safe for concurrent mutation, but all MLX FFI use in this crate is
// serialized by `backend_metal.rs`'s `MLX_GUARD`, which is the synchronization
// boundary for moving these opaque handles across threads.
unsafe impl Send for MlxHandle {}

#[cfg(feature = "metal")]
// Safety: see the `Send` impl above. Shared references are only used to pass
// opaque handles into MLX while holding `MLX_GUARD`.
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

    pub(crate) fn len(&self) -> usize {
        self.inner.len()
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

pub trait Backend: std::fmt::Debug + Send + Sync {
    fn device(&self) -> Device;

    fn upload(&self, host: &[f32], _shape: &[usize]) -> Result<DeviceHandle> {
        Ok(DeviceHandle::Cpu(host.to_vec()))
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

    /// Elementwise `C = A + B` over identically-shaped contiguous tensors.
    /// Lazy on backends that support it (e.g. Metal defers to `mlx_eval`).
    fn add(&self, a: &DeviceHandle, b: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle>;

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

    /// Elementwise `out = a * b` over identically-sized contiguous tensors.
    fn mul_forward(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        cpu_mul_forward(a, b)
    }

    /// Elementwise `out = a * s` for scalar `s`.
    fn mul_scalar_forward(&self, a: &[f32], s: f32) -> Result<Vec<f32>> {
        cpu_mul_scalar_forward(a, s)
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

    /// Rotary position embedding (NeoX / `rotate_half` layout, matches Qwen3).
    /// `x` is `[batch, heads, seq, head_dim]`; `cos`/`sin` are `[seq, head_dim/2]`.
    /// Returns the rotated tensor with the same shape as `x`.
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
            for row in 0..m {
                for col in 0..n {
                    let mut acc = 0.0f32;
                    for inner in 0..k {
                        acc += a[(row * k) + inner] * b[(inner * n) + col];
                    }
                    out[(row * n) + col] = acc;
                }
            }
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
                for row in 0..m {
                    for col in 0..n {
                        let mut acc = 0.0f32;
                        for inner in 0..k {
                            acc += a[a_base + (row * k) + inner] * b[b_base + (inner * n) + col];
                        }
                        out[out_base + (row * n) + col] = acc;
                    }
                }
            }
            Ok((out, out_shape))
        }
        _ => Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2 or rank-3",
            got: a_shape.len().max(b_shape.len()),
        }),
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
    let expected_out = matmul_output_shape(a_shape, b_shape)?;
    if grad_out_shape != expected_out.as_slice() {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: expected_out,
            got: grad_out_shape.to_vec(),
        });
    }

    let grad_a = if need_grad_a {
        // grad_a = grad_out @ b^T
        let (b_t, b_t_shape) = transpose_last_two_ref(b, b_shape);
        let (data, _) = cpu_matmul_forward(grad_out, grad_out_shape, &b_t, &b_t_shape)?;
        data
    } else {
        Vec::new()
    };
    let grad_b = if need_grad_b {
        // grad_b = a^T @ grad_out
        let (a_t, a_t_shape) = transpose_last_two_ref(a, a_shape);
        let (data, _) = cpu_matmul_forward(&a_t, &a_t_shape, grad_out, grad_out_shape)?;
        data
    } else {
        Vec::new()
    };
    Ok((grad_a, grad_b))
}

/// Transpose the inner-most two axes of a rank-2 or rank-3 row-major buffer.
/// Pure-host scratch used by `cpu_matmul_backward` and the `no-cuda`
/// type-check path of the CUDA backend.
pub(crate) fn transpose_last_two_ref(data: &[f32], shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    match shape.len() {
        2 => {
            let rows = shape[0];
            let cols = shape[1];
            let mut out = vec![0.0f32; rows * cols];
            for row in 0..rows {
                for col in 0..cols {
                    out[col * rows + row] = data[row * cols + col];
                }
            }
            (out, vec![cols, rows])
        }
        3 => {
            let batch = shape[0];
            let rows = shape[1];
            let cols = shape[2];
            let plane = rows * cols;
            let mut out = vec![0.0f32; batch * plane];
            for batch_index in 0..batch {
                let base = batch_index * plane;
                for row in 0..rows {
                    for col in 0..cols {
                        out[base + col * rows + row] = data[base + row * cols + col];
                    }
                }
            }
            (out, vec![batch, cols, rows])
        }
        _ => (data.to_vec(), shape.to_vec()),
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
    let expected_cache = seq * half_dim;
    if cos.len() != expected_cache || sin.len() != expected_cache {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected_cache],
            got: vec![cos.len().min(sin.len())],
        });
    }
    let mut out = vec![0.0_f32; expected_x];
    for b in 0..batch {
        for h in 0..heads {
            for t in 0..seq {
                let rope_base = t * half_dim;
                let base = (((b * heads) + h) * seq + t) * head_dim;
                for i in 0..half_dim {
                    let x0 = x[base + i];
                    let x1 = x[base + i + half_dim];
                    let c = cos[rope_base + i];
                    let s = sin[rope_base + i];
                    out[base + i] = (x0 * c) - (x1 * s);
                    out[base + i + half_dim] = (x1 * c) + (x0 * s);
                }
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
