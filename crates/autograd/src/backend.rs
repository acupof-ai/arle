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

    /// Elementwise `C = A + B` over identically-shaped contiguous tensors.
    /// Lazy on backends that support it (e.g. Metal defers to `mlx_eval`).
    fn add(&self, a: &DeviceHandle, b: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle>;

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
