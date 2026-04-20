//! Metal backend via mlx-sys. Host `Vec<f32>` stays authoritative: upload
//! into `mlx_array`, call `mlx_matmul`, `mlx_eval`, copy the result back.
//!
//! MLX's `mx::matmul` natively supports batched (rank-3) row-major inputs,
//! so we pass shape through unchanged. Shape validation mirrors
//! `cpu_matmul_forward` so the trait contract stays identical.

use crate::{
    AutogradError, Result,
    backend::{Backend, Device, DeviceHandle, MlxHandle, matmul_output_shape, validate_broadcast},
};
use mlx_sys::{
    MLX_FLOAT32, MLX_INT32, mlx_add, mlx_array, mlx_array_data_float32, mlx_array_free,
    mlx_array_from_data, mlx_array_new_float32, mlx_array_size, mlx_concatenate_axis,
    mlx_contiguous, mlx_erf, mlx_eval, mlx_exp, mlx_fast_rms_norm, mlx_logsumexp_axis, mlx_matmul,
    mlx_mean_axis, mlx_multiply, mlx_negative, mlx_reciprocal, mlx_reshape,
    mlx_scatter_add_rows_f32, mlx_sigmoid, mlx_slice, mlx_softmax_axis, mlx_sqrt, mlx_subtract,
    mlx_sum_axis, mlx_take_axis, mlx_tanh, mlx_transpose_axes,
};
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

// MLX's default stream/device is process-global and its C++ allocator is
// not re-entrant across threads. Concurrent `mlx_matmul` calls (e.g.
// default `cargo test` parallelism) SEGV the interpreter. A static
// mutex here is coarse but correct — training is single-threaded, and
// the lock is held only for the duration of one matmul FFI round-trip.
pub(crate) static MLX_GUARD: Mutex<()> = Mutex::new(());

// Per-process counter for every `mlx_eval` call that flows through the
// Metal backend. Used by M5.3a acceptance tests to confirm that a
// well-structured forward+backward step terminates in exactly one eval
// boundary (see `docs/plans/m5.3-device-resident-tensor.md` §5). Covers
// `MetalBackend::eval` as well as the legacy `eval_and_readback` tail
// used by non-device-resident ops.
static METAL_EVAL_COUNT: AtomicU64 = AtomicU64::new(0);

/// Number of `mlx_eval` invocations performed by the Metal backend since
/// process start (or the last `reset_eval_count`). Cheap atomic load;
/// intended for tests and bench harnesses, not hot-path instrumentation.
pub fn eval_count() -> u64 {
    METAL_EVAL_COUNT.load(Ordering::Relaxed)
}

/// Reset the eval counter to zero. Exposed so test harnesses can scope a
/// measurement around a single training step. Thread-safe but not
/// synchronized with concurrent `mlx_eval` calls — the training tape is
/// single-threaded, so resetting immediately before the step under test
/// is the contract.
pub fn reset_eval_count() {
    METAL_EVAL_COUNT.store(0, Ordering::Relaxed);
}

#[inline]
fn bump_eval_count() {
    METAL_EVAL_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MetalBackend;

impl Backend for MetalBackend {
    fn device(&self) -> Device {
        Device::Metal
    }

    fn upload(&self, host: &[f32], shape: &[usize]) -> Result<DeviceHandle> {
        let shape_i32: Vec<i32> = shape.iter().map(|&dim| dim as i32).collect();
        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: `host` and `shape_i32` stay alive for the duration of the FFI
        // call, MLX copies from the host slice into its own array storage, and
        // the returned pointer becomes uniquely owned by the `MlxHandle`.
        let handle = unsafe {
            let array = mlx_array_from_data(
                host.as_ptr() as *const c_void,
                shape_i32.as_ptr(),
                shape_i32.len() as i32,
                MLX_FLOAT32,
            );
            if array.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_array_from_data returned null",
                ));
            }
            DeviceHandle::Metal(MlxHandle::from_raw(array))
        };

        Ok(handle)
    }

    fn readback(&self, handle: &DeviceHandle) -> Result<Vec<f32>> {
        match handle {
            DeviceHandle::Cpu(data) => Ok(data.clone()),
            DeviceHandle::Metal(handle) => {
                let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

                // Safety: the raw MLX array pointer is owned by `handle` for the
                // duration of this borrow, the caller is responsible for having
                // evaluated the array before readback, and the destination host
                // buffer is freshly allocated for this copy.
                let host = unsafe {
                    let array = handle.as_ptr();
                    let size = mlx_array_size(array);
                    let data_ptr = mlx_array_data_float32(array);
                    if data_ptr.is_null() {
                        return Err(AutogradError::TapeInvariant(
                            "mlx_array_data_float32 returned null",
                        ));
                    }

                    let mut out = vec![0.0f32; size];
                    std::ptr::copy_nonoverlapping(data_ptr, out.as_mut_ptr(), size);
                    out
                };

                Ok(host)
            }
            #[cfg(feature = "cuda")]
            DeviceHandle::Cuda(_) => Err(AutogradError::TapeInvariant(
                "metal backend cannot read back a cuda device handle",
            )),
        }
    }

    fn eval(&self, handles: &[&DeviceHandle]) -> Result<()> {
        let mut metal_handles = handles
            .iter()
            .filter_map(|handle| match handle {
                DeviceHandle::Metal(handle) => Some(handle.as_ptr()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if metal_handles.is_empty() {
            return Ok(());
        }

        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: each pointer comes from a live `MlxHandle` borrowed for the
        // duration of this call, ownership stays with those handles, and MLX
        // access is serialized under `MLX_GUARD`.
        unsafe {
            mlx_eval(metal_handles.as_mut_ptr(), metal_handles.len());
        }
        bump_eval_count();

        Ok(())
    }

    fn matmul(
        &self,
        a: &DeviceHandle,
        a_shape: &[usize],
        b: &DeviceHandle,
        b_shape: &[usize],
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        let out_shape = matmul_output_shape(a_shape, b_shape)?;
        let DeviceHandle::Metal(a_handle) = a else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot matmul a non-metal device handle",
            ));
        };
        let DeviceHandle::Metal(b_handle) = b else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot matmul a non-metal device handle",
            ));
        };

        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: both pointers come from live `MlxHandle`s borrowed for this
        // call, ownership of the returned MLX node transfers into the new
        // `MlxHandle`, and `MLX_GUARD` serializes access to MLX's global state.
        let out = unsafe {
            let out_arr = mlx_matmul(a_handle.as_ptr(), b_handle.as_ptr());
            if out_arr.is_null() {
                return Err(AutogradError::TapeInvariant("mlx_matmul returned null"));
            }
            DeviceHandle::Metal(MlxHandle::from_raw(out_arr))
        };

        Ok((out, out_shape))
    }

    fn matmul_forward(
        &self,
        a: &[f32],
        a_shape: &[usize],
        b: &[f32],
        b_shape: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let a_handle = self.upload(a, a_shape)?;
        let b_handle = self.upload(b, b_shape)?;
        let (out_handle, out_shape) = self.matmul(&a_handle, a_shape, &b_handle, b_shape)?;
        self.eval(&[&out_handle])?;
        let out = self.readback(&out_handle)?;
        let expected_size: usize = out_shape.iter().product();
        if out.len() != expected_size {
            return Err(AutogradError::ShapeMismatch {
                expected: out_shape.clone(),
                got: vec![out.len()],
            });
        }
        Ok((out, out_shape))
    }

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
        mlx_matmul_backward(
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

    fn softmax_forward_last_axis(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
        mlx_softmax_like(x, shape, SoftmaxKind::Softmax)
    }

    fn log_softmax_forward_last_axis(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
        mlx_softmax_like(x, shape, SoftmaxKind::LogSoftmax)
    }

    fn add(&self, a: &DeviceHandle, b: &DeviceHandle, _shape: &[usize]) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(a_handle) = a else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot add a non-metal device handle",
            ));
        };
        let DeviceHandle::Metal(b_handle) = b else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot add a non-metal device handle",
            ));
        };

        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: both pointers come from live `MlxHandle`s borrowed for this
        // call, ownership of the returned MLX node transfers into the new
        // `MlxHandle`, and `MLX_GUARD` serializes access to MLX's global state.
        let out = unsafe {
            let out_arr = mlx_add(a_handle.as_ptr(), b_handle.as_ptr());
            if out_arr.is_null() {
                return Err(AutogradError::TapeInvariant("mlx_add returned null"));
            }
            DeviceHandle::Metal(MlxHandle::from_raw(out_arr))
        };

        Ok(out)
    }

    // Lazy right-aligned broadcast-add: MLX's `mlx_add` already implements
    // NumPy-style right-aligned broadcasting, so the lazy path is just the
    // same FFI call as `add` — no explicit `mlx_broadcast_to` needed.
    // `a_shape`/`b_shape` are passed through for the host-fallback contract
    // but ignored on Metal (MLX reads shapes off the arrays themselves).
    // Output shape equals `a_shape`. M5.3b.14.
    fn add_broadcast(
        &self,
        a: &DeviceHandle,
        _a_shape: &[usize],
        b: &DeviceHandle,
        _b_shape: &[usize],
    ) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(a_handle) = a else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot add_broadcast a non-metal device handle",
            ));
        };
        let DeviceHandle::Metal(b_handle) = b else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot add_broadcast a non-metal device handle",
            ));
        };

        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: both pointers come from live `MlxHandle`s borrowed for
        // this call; ownership of the returned node transfers into the
        // new `MlxHandle`; `MLX_GUARD` serializes access to MLX globals.
        let out = unsafe {
            let out_arr = mlx_add(a_handle.as_ptr(), b_handle.as_ptr());
            if out_arr.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_add returned null (add_broadcast)",
                ));
            }
            DeviceHandle::Metal(MlxHandle::from_raw(out_arr))
        };

        Ok(out)
    }

    // Lazy reduce-sum-all: reshape `x` into a 1-D `[N]` view (an MLX no-op
    // when the input is contiguous) and call `mlx_sum_axis(_, 0, keepdims=false)`
    // to produce a rank-0 scalar that composes into MLX's lazy graph. NO
    // `mlx_eval` here — the tape's terminal flush (`ensure_host` on the loss)
    // is the single eval boundary. This is the M5.3b.1 deliverable; before
    // M5.3b.1 `sum` always forced a flush via `sum_last_axis_forward`.
    fn sum_all(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(x_handle) = x else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot sum a non-metal device handle",
            ));
        };
        let size = if shape.is_empty() {
            1
        } else {
            shape.iter().product::<usize>()
        };

        let flat_shape = [size as i32];
        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: `x_handle` is a live MLX array borrowed for this call;
        // both intermediate arrays we allocate here are freed (the reshape
        // node) or transferred into `MlxHandle` (the sum result) before
        // returning, and `MLX_GUARD` serializes MLX state access.
        let out = unsafe {
            let flat = mlx_reshape(x_handle.as_ptr(), flat_shape.as_ptr(), 1);
            if flat.is_null() {
                return Err(AutogradError::TapeInvariant("mlx_reshape returned null"));
            }
            let out_arr = mlx_sum_axis(flat, 0, false);
            mlx_array_free(flat);
            if out_arr.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_sum_axis returned null (sum_all)",
                ));
            }
            DeviceHandle::Metal(MlxHandle::from_raw(out_arr))
        };

        Ok(out)
    }

    // Lazy row-wise softmax over the last axis. Composes
    // `mlx_softmax_axis(x, -1, keepdims=true)` into the MLX graph with
    // no `mlx_eval` — the tape's terminal flush is the single eval
    // boundary. Mirrors the eager `mlx_softmax_like` path but skips the
    // upload-from-host + eval+readback tail since `x` is already an MLX
    // array. M5.3b.2.
    fn softmax_last_axis(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(x_handle) = x else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot softmax a non-metal device handle",
            ));
        };
        validate_softmax_shape(shape)?;

        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: `x_handle` is a live MLX array borrowed for this call;
        // `mlx_softmax_axis` returns a fresh node that we transfer into a
        // new `MlxHandle`. `MLX_GUARD` serializes MLX state access.
        let out = unsafe {
            let out_arr = mlx_softmax_axis(x_handle.as_ptr(), -1_i32, true);
            if out_arr.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_softmax_axis returned null (softmax_last_axis)",
                ));
            }
            DeviceHandle::Metal(MlxHandle::from_raw(out_arr))
        };

        Ok(out)
    }

    // Lazy elementwise SiLU: composes `x * mlx_sigmoid(x)` into the MLX
    // graph with no `mlx_eval`. The intermediate `sig` node is freed after
    // the multiply; the returned handle owns the multiply result. Shape is
    // passed through (unused on the MLX side — the output broadcasts to
    // the input shape automatically). M5.3b.3.
    fn silu(&self, x: &DeviceHandle, _shape: &[usize]) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(x_handle) = x else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot silu a non-metal device handle",
            ));
        };

        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: `x_handle` is a live MLX array borrowed for this call;
        // `sig` is allocated here and freed before return; the `out`
        // result is transferred into the returned `MlxHandle`.
        let out = unsafe {
            let sig = mlx_sigmoid(x_handle.as_ptr());
            if sig.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_sigmoid returned null (silu)",
                ));
            }
            let out_arr = mlx_multiply(x_handle.as_ptr(), sig);
            mlx_array_free(sig);
            if out_arr.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_multiply returned null (silu)",
                ));
            }
            DeviceHandle::Metal(MlxHandle::from_raw(out_arr))
        };

        Ok(out)
    }

    // Lazy elementwise exp: pipes `x` through `mlx_exp` with no
    // `mlx_eval`. The returned handle owns the exp result. Shape is
    // passed through (unused on the MLX side — output broadcasts to the
    // input shape). M5.3b.4.
    fn exp(&self, x: &DeviceHandle, _shape: &[usize]) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(x_handle) = x else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot exp a non-metal device handle",
            ));
        };

        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: `x_handle` is a live MLX array borrowed for this call;
        // the `out_arr` result is transferred into the returned `MlxHandle`.
        let out = unsafe {
            let out_arr = mlx_exp(x_handle.as_ptr());
            if out_arr.is_null() {
                return Err(AutogradError::TapeInvariant("mlx_exp returned null (exp)"));
            }
            DeviceHandle::Metal(MlxHandle::from_raw(out_arr))
        };

        Ok(out)
    }

    // Lazy elementwise sigmoid: pipes `x` through `mlx_sigmoid` with no
    // `mlx_eval`. The returned handle owns the sigmoid result. Shape is
    // passed through (unused on the MLX side — output matches the input
    // shape). Qwen3.5 attention gates `gate = sigmoid(gate_proj)` once per
    // layer × 28 layers. M5.3b.18.
    fn sigmoid(&self, x: &DeviceHandle, _shape: &[usize]) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(x_handle) = x else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot sigmoid a non-metal device handle",
            ));
        };

        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: `x_handle` is a live MLX array borrowed for this call;
        // the `out_arr` result is transferred into the returned `MlxHandle`.
        let out = unsafe {
            let out_arr = mlx_sigmoid(x_handle.as_ptr());
            if out_arr.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_sigmoid returned null (sigmoid)",
                ));
            }
            DeviceHandle::Metal(MlxHandle::from_raw(out_arr))
        };

        Ok(out)
    }

    // Lazy scalar-broadcast multiply: composes `mlx_multiply(x, scalar)`
    // into the MLX graph with no `mlx_eval`. The scalar is allocated as a
    // rank-0 `mlx_array` via `mlx_array_new_float32` and freed after the
    // multiply; MLX broadcasts rank-0 scalars across any rank. Shape is
    // passed through (unused — output matches input shape). M5.3b.13.
    // M5.3b.17: elementwise `a * b` via `mlx_multiply`. Hot-path in
    // Qwen3.5: `attn * gate` (sigmoid-gated attention, 1 per layer) and
    // `silu(gate) * up` (MLP SwiGLU activation, 1 per layer). Shapes must
    // match on both sides (caller validates; MLX's `mlx_multiply` actually
    // broadcasts right-aligned but this wrapper is for the elementwise
    // `ops::mul` which keeps shape equality as a precondition).
    fn mul(&self, a: &DeviceHandle, b: &DeviceHandle, _shape: &[usize]) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(a_handle) = a else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot mul a non-metal lhs handle",
            ));
        };
        let DeviceHandle::Metal(b_handle) = b else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot mul a non-metal rhs handle",
            ));
        };

        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");
        unsafe {
            let out = mlx_multiply(a_handle.as_ptr(), b_handle.as_ptr());
            if out.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_multiply returned null (elementwise mul)",
                ));
            }
            Ok(DeviceHandle::Metal(MlxHandle::from_raw(out)))
        }
    }

    fn mul_scalar(&self, x: &DeviceHandle, s: f32, _shape: &[usize]) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(x_handle) = x else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot mul_scalar a non-metal device handle",
            ));
        };

        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: `x_handle` is a live MLX array borrowed for this call;
        // `scalar` is allocated here and freed before return; the `out`
        // result is transferred into the returned `MlxHandle`.
        let out = unsafe {
            let scalar = mlx_array_new_float32(s);
            if scalar.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_array_new_float32 returned null (mul_scalar)",
                ));
            }
            let out_arr = mlx_multiply(x_handle.as_ptr(), scalar);
            mlx_array_free(scalar);
            if out_arr.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_multiply returned null (mul_scalar)",
                ));
            }
            DeviceHandle::Metal(MlxHandle::from_raw(out_arr))
        };

        Ok(out)
    }

    // Lazy row-wise log-softmax over the last axis. Composes
    // `x - mlx_logsumexp_axis(x, -1, keepdims=true)` into the MLX graph
    // with no `mlx_eval`. The intermediate `lse` node is freed after the
    // subtract; the returned handle owns the subtract result. M5.3b.2.
    fn log_softmax_last_axis(&self, x: &DeviceHandle, shape: &[usize]) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(x_handle) = x else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot log_softmax a non-metal device handle",
            ));
        };
        validate_softmax_shape(shape)?;

        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: `x_handle` is a live MLX array borrowed for this call;
        // `lse` is allocated here and freed before return; the `diff`
        // result is transferred into the returned `MlxHandle`.
        let out = unsafe {
            let lse = mlx_logsumexp_axis(x_handle.as_ptr(), -1_i32, true);
            if lse.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_logsumexp_axis returned null (log_softmax_last_axis)",
                ));
            }
            let diff = mlx_subtract(x_handle.as_ptr(), lse);
            mlx_array_free(lse);
            if diff.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_subtract returned null (log_softmax_last_axis)",
                ));
            }
            DeviceHandle::Metal(MlxHandle::from_raw(diff))
        };

        Ok(out)
    }

    fn mul_forward(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![a.len()],
                got: vec![b.len()],
            });
        }
        mlx_binary_flat(a, b, BinaryOp::Mul)
    }

    fn mul_scalar_forward(&self, a: &[f32], s: f32) -> Result<Vec<f32>> {
        mlx_unary_flat(a, UnaryOp::MulScalar(s))
    }

    fn add_broadcast_forward(
        &self,
        a: &[f32],
        a_shape: &[usize],
        b: &[f32],
        b_shape: &[usize],
    ) -> Result<Vec<f32>> {
        mlx_add_broadcast(a, a_shape, b, b_shape)
    }

    fn exp_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        mlx_unary_flat(a, UnaryOp::Exp)
    }

    fn neg_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        mlx_unary_flat(a, UnaryOp::Neg)
    }

    fn gelu_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        mlx_unary_flat(a, UnaryOp::Gelu)
    }

    fn silu_forward(&self, a: &[f32]) -> Result<Vec<f32>> {
        mlx_unary_flat(a, UnaryOp::Silu)
    }

    fn rms_norm_forward(
        &self,
        x: &[f32],
        weight: &[f32],
        shape: &[usize],
        eps: f32,
    ) -> Result<Vec<f32>> {
        mlx_rms_norm(x, weight, shape, eps)
    }

    fn embedding_forward(
        &self,
        weight: &[f32],
        vocab: usize,
        dim: usize,
        ids: &[i32],
    ) -> Result<Vec<f32>> {
        mlx_embedding(weight, vocab, dim, ids)
    }

    fn sum_last_axis_forward(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
        mlx_reduce_last_axis(x, shape, ReduceOp::Sum)
    }

    fn mean_last_axis_forward(&self, x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
        mlx_reduce_last_axis(x, shape, ReduceOp::Mean)
    }

    fn rope_forward(
        &self,
        x: &[f32],
        x_shape: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<Vec<f32>> {
        mlx_rope(x, x_shape, cos, sin)
    }

    // Lazy fused row-wise RMSNorm: borrows `x` as an existing MLX handle,
    // uploads `weight` per call (typically host-resident), and returns
    // the `mlx_fast_rms_norm` output wrapped in an `MlxHandle` without
    // evaluating. Backward recomputes `inv_rms` host-side from the saved
    // `x` (flushed to host by `tape.backward` before backward walks).
    // M5.3b.6.
    fn rms_norm(
        &self,
        x: &DeviceHandle,
        weight: &[f32],
        shape: &[usize],
        eps: f32,
    ) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(x_handle) = x else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot rms_norm a non-metal device handle",
            ));
        };
        mlx_rms_norm_lazy(x_handle.as_ptr(), weight, shape, eps)
    }

    // Lazy half-split rotation: same graph as `mlx_rope` (slice → multiply →
    // sub/add → concat) but borrows `x` as an existing MLX handle and skips
    // the final `eval_and_readback`. cos/sin still upload per call from host
    // slices — the Qwen caches are precomputed per seq length and rarely
    // benefit from staying device-resident, and this keeps the API from
    // needing to merge three device handles. M5.3b.5.
    fn rope(
        &self,
        x: &DeviceHandle,
        x_shape: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(x_handle) = x else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot rope a non-metal device handle",
            ));
        };
        mlx_rope_lazy(x_handle.as_ptr(), x_shape, cos, sin)
    }

    // Lazy erf-based GELU: `0.5 * x * (1 + erf(x * INV_SQRT_2))`. Matches
    // `ops::activation::gelu`'s CPU inline formula (NOT the tanh-approx
    // `gelu_forward` on the trait). `gelu_backward` uses the erf-derivative
    // against the saved input tensor, so the lazy forward must stay on
    // the erf form or forward/backward become inconsistent by ~1e-3 per
    // element. M5.3b.8.
    fn gelu(&self, x: &DeviceHandle, _shape: &[usize]) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(x_handle) = x else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot gelu a non-metal device handle",
            ));
        };
        mlx_gelu_erf_lazy(x_handle.as_ptr())
    }

    // Lazy embedding gather: upload the tiny `[seq]` int32 id array per
    // call (no benefit to caching — ids change every step), `mlx_take_axis`
    // along axis 0 to pick the rows, then `mlx_reshape` from `[seq, hidden]`
    // to `[1, seq, hidden]` to match `ops::embedding`'s batch-row
    // convention. No eval — the whole sequence composes into the MLX
    // graph. Backward stays on the host scatter-add path (already
    // `mlx_scatter_add_rows_f32`-backed). M5.3b.7.
    fn embedding(
        &self,
        table: &DeviceHandle,
        table_shape: &[usize],
        ids: &[i32],
    ) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(table_handle) = table else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot embedding a non-metal device handle",
            ));
        };
        mlx_embedding_lazy(table_handle.as_ptr(), table_shape, ids)
    }

    fn gather_last_dim_forward(
        &self,
        src: &[f32],
        src_shape: &[usize],
        ids: &[i32],
    ) -> Result<Vec<f32>> {
        mlx_gather_last_dim(src, src_shape, ids)
    }

    // Lazy gather along the last axis. `src` is borrowed as a live MLX
    // handle; flatten to `[prefix*vocab]`, upload the remapped flat ids
    // (`i * vocab + ids[i]`) as a tiny int32 array, `mlx_take_axis(axis=0)`
    // picks the chosen element per row, final `mlx_reshape` restores
    // `src_shape[..-1]`. No eval — the whole chain composes into the MLX
    // graph. Backward stays on the host scatter-add path (already
    // `mlx_scatter_add_rows_f32`-backed). M5.3b.9.
    fn gather_last_dim(
        &self,
        src: &DeviceHandle,
        src_shape: &[usize],
        ids: &[i32],
    ) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(src_handle) = src else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot gather_last_dim a non-metal device handle",
            ));
        };
        mlx_gather_last_dim_lazy(src_handle.as_ptr(), src_shape, ids)
    }

    fn scatter_add_rows_forward(
        &self,
        upstream: &[f32],
        prefix_rows: usize,
        feature_dim: usize,
        indices: &[i32],
        vocab: usize,
    ) -> Result<Vec<f32>> {
        mlx_scatter_add_rows(upstream, prefix_rows, feature_dim, indices, vocab)
    }

    // M5.3b.12: reshape is a pure metadata op on MLX — `mlx_reshape` composes
    // into the lazy graph without triggering an eval. No new MLX primitives.
    // Stripped `ensure_host` at ops/layout.rs → reshape now stays device-
    // resident through q/k/v prep in every Qwen3.5 attention layer.
    fn reshape(&self, x: &DeviceHandle, new_shape: &[usize]) -> Result<DeviceHandle> {
        let DeviceHandle::Metal(x_handle) = x else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot reshape a non-metal device handle",
            ));
        };
        let shape_i32: Vec<i32> = new_shape.iter().map(|&d| d as i32).collect();
        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");
        unsafe {
            let reshaped = mlx_reshape(x_handle.as_ptr(), shape_i32.as_ptr(), shape_i32.len());
            if reshaped.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_reshape returned null (device reshape)",
                ));
            }
            Ok(DeviceHandle::Metal(MlxHandle::from_raw(reshaped)))
        }
    }

    // M5.3b.12: transpose is also free on MLX (`mlx_transpose_axes` creates a
    // lazy view), so the whole attention-prep chain
    // `matmul → reshape → slice → transpose → rope → matmul`
    // now stays on the lazy graph. Permutation is identity except axis1↔axis2;
    // MLX fuses the view into downstream GEMMs.
    fn transpose_axes_swap(
        &self,
        x: &DeviceHandle,
        old_shape: &[usize],
        axis1: usize,
        axis2: usize,
    ) -> Result<(DeviceHandle, Vec<usize>)> {
        let rank = old_shape.len();
        if axis1 >= rank {
            return Err(AutogradError::AxisOutOfBounds { axis: axis1, rank });
        }
        if axis2 >= rank {
            return Err(AutogradError::AxisOutOfBounds { axis: axis2, rank });
        }
        let DeviceHandle::Metal(x_handle) = x else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot transpose a non-metal device handle",
            ));
        };
        let mut new_shape = old_shape.to_vec();
        new_shape.swap(axis1, axis2);
        if axis1 == axis2 {
            // Identity: return a reshape-same-shape clone, matches the
            // non-swap fast path in `cpu_transpose_swap`. `mlx_reshape`
            // with identical shape is a cheap view alias; avoids a no-op
            // permutation and keeps ownership semantics consistent.
            let shape_i32: Vec<i32> = new_shape.iter().map(|&d| d as i32).collect();
            let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");
            let view =
                unsafe { mlx_reshape(x_handle.as_ptr(), shape_i32.as_ptr(), shape_i32.len()) };
            if view.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_reshape returned null (transpose axis1==axis2 alias)",
                ));
            }
            return Ok((DeviceHandle::Metal(MlxHandle::from_raw(view)), new_shape));
        }
        // Build identity permutation then swap the two chosen axes.
        let mut perm: Vec<i32> = (0..rank as i32).collect();
        perm.swap(axis1, axis2);
        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");
        unsafe {
            let view = mlx_transpose_axes(x_handle.as_ptr(), perm.as_ptr(), perm.len());
            if view.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_transpose_axes returned null (device transpose)",
                ));
            }
            // `mlx_transpose_axes` returns a lazy non-contiguous VIEW; the
            // raw `mlx_array_data_float32` pointer readback uses ignores
            // strides and yields the original layout, causing a silent
            // row-ordering parity bug against CPU. Wrap in `mlx_contiguous`
            // so the returned handle materializes in the new layout on the
            // next eval. MLX short-circuits `contiguous` on already-contig
            // arrays, so the cost is zero in the common case; the copy
            // only fires when the view IS non-contiguous (i.e. always,
            // for a non-identity swap — but only once per logical op,
            // the way the pre-M5.3b.12 host-path always produced
            // contiguous output).
            let out = mlx_contiguous(view);
            mlx_array_free(view);
            if out.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_contiguous returned null (device transpose materialization)",
                ));
            }
            Ok((DeviceHandle::Metal(MlxHandle::from_raw(out)), new_shape))
        }
    }

    // M5.3b.16: `mlx_slice` is a lazy view op — fuses with downstream GEMMs
    // when the slice feeds a matmul (as in Qwen3.5's q/gate split per
    // attention layer). Strides are all 1 since autograd's `slice` is
    // always a contiguous window. Same view-materialization trap as
    // transpose: the raw `mlx_array_data_float32` pointer ignores view
    // strides/offsets, so wrap the result in `mlx_contiguous` to force
    // a layout-correct materialization on the next eval. MLX short-
    // circuits `contiguous` on already-contig arrays; the copy only
    // fires when the view is non-contiguous (always, for a non-full
    // slice — but only once per logical op).
    fn slice(
        &self,
        x: &DeviceHandle,
        old_shape: &[usize],
        starts: &[usize],
        ends: &[usize],
    ) -> Result<DeviceHandle> {
        let rank = old_shape.len();
        if starts.len() != rank || ends.len() != rank {
            return Err(AutogradError::InvalidIndicesLen {
                expected: rank,
                got: starts.len().max(ends.len()),
            });
        }
        for ((&s, &e), &d) in starts.iter().zip(ends.iter()).zip(old_shape.iter()) {
            if s > e {
                return Err(AutogradError::TapeInvariant(
                    "metal slice: start must be <= end for every axis",
                ));
            }
            if e > d {
                return Err(AutogradError::IndexOutOfBounds { index: e, upper: d });
            }
        }
        let DeviceHandle::Metal(x_handle) = x else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot slice a non-metal device handle",
            ));
        };

        let starts_i32: Vec<i32> = starts.iter().map(|&s| s as i32).collect();
        let ends_i32: Vec<i32> = ends.iter().map(|&e| e as i32).collect();
        let strides_i32: Vec<i32> = vec![1; rank];

        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");
        unsafe {
            let view = mlx_slice(
                x_handle.as_ptr(),
                starts_i32.as_ptr(),
                ends_i32.as_ptr(),
                strides_i32.as_ptr(),
                rank,
            );
            if view.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_slice returned null (device slice)",
                ));
            }
            let out = mlx_contiguous(view);
            mlx_array_free(view);
            if out.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_contiguous returned null (device slice materialization)",
                ));
            }
            Ok(DeviceHandle::Metal(MlxHandle::from_raw(out)))
        }
    }

    // Lazy fused AdamW per-param update. Upload `grad` once as a flat
    // `[size]` f32 array and compose the entire update into the MLX graph:
    //
    //   m' = β1·m + (1-β1)·g
    //   v' = β2·v + (1-β2)·g²
    //   param' = (1 - lr·wd)·param - lr·(m'/bc1) / (√(v'/bc2) + eps)
    //
    // M5.3b.11: returns the three MLX graph nodes UNEVALUATED. The caller
    // (`AdamW::step_device`) collects every param's triple and issues a
    // single `backend.eval(&handles)` at the end of the optimizer step,
    // so per-step eval count is `1` regardless of param count (~200 on
    // Qwen3.5-class models). Composing independent per-param chains into
    // one eval is safe — the updates share no sub-node.
    //
    // M5.3b.10 context (preserved for reference): the prior host-loop path
    // downloaded + uploaded every param every step (`get_mut` →
    // `ensure_host` → mutate → mark Dirty::Host → next `ensure_device`
    // re-uploads). Staying Dirty::Device across steps kills that churn.
    // Scalar constants broadcast-multiply via `mlx_array_new_float32`,
    // the same primitive the lazy `gelu` uses for 0.5/INV_SQRT_2. No new
    // MLX primitives introduced — `mlx_divide` does not exist in the
    // bridge, so we reach the reciprocal via `mlx_reciprocal`.
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
        let size: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        if grad.len() != size {
            return Err(AutogradError::DataLengthMismatch {
                len: grad.len(),
                shape: shape.to_vec(),
                size,
            });
        }

        let DeviceHandle::Metal(param_handle) = param else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot adamw_step a non-metal param handle",
            ));
        };
        let DeviceHandle::Metal(m_handle) = m else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot adamw_step a non-metal m handle",
            ));
        };
        let DeviceHandle::Metal(v_handle) = v else {
            return Err(AutogradError::TapeInvariant(
                "metal backend cannot adamw_step a non-metal v handle",
            ));
        };

        let shape_i32: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: every intermediate array we allocate is either freed
        // before return or transferred into an `MlxHandle` at the final
        // wrap site. `param_handle` / `m_handle` / `v_handle` are borrowed
        // for this whole call under `MLX_GUARD`; the returned handles own
        // fresh MLX arrays that become the caller's new params+moments.
        // We run the composition inside a single closure so `?` cleanup
        // drops intermediates correctly — but every step below allocates
        // arrays that we must explicitly free via `mlx_array_free` if the
        // next allocation fails. Follows the same free-on-null pattern as
        // `mlx_gelu_erf_lazy` in this file.
        unsafe {
            // Upload grad as a flat [size] f32 array.
            let grad_arr = mlx_array_from_data(
                grad.as_ptr() as *const c_void,
                shape_i32.as_ptr(),
                shape_i32.len() as i32,
                MLX_FLOAT32,
            );
            if grad_arr.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_array_from_data returned null (adamw grad)",
                ));
            }

            // Scalar constants for the composition.
            let beta1_arr = mlx_array_new_float32(beta1);
            let one_minus_beta1 = mlx_array_new_float32(1.0_f32 - beta1);
            let beta2_arr = mlx_array_new_float32(beta2);
            let one_minus_beta2 = mlx_array_new_float32(1.0_f32 - beta2);
            let inv_bc1 = mlx_array_new_float32(1.0_f32 / bc1);
            let inv_bc2 = mlx_array_new_float32(1.0_f32 / bc2);
            let eps_arr = mlx_array_new_float32(eps);
            let lr_arr = mlx_array_new_float32(lr);
            let decay_arr = mlx_array_new_float32(1.0_f32 - (lr * wd));
            if beta1_arr.is_null()
                || one_minus_beta1.is_null()
                || beta2_arr.is_null()
                || one_minus_beta2.is_null()
                || inv_bc1.is_null()
                || inv_bc2.is_null()
                || eps_arr.is_null()
                || lr_arr.is_null()
                || decay_arr.is_null()
            {
                mlx_array_free(grad_arr);
                if !beta1_arr.is_null() {
                    mlx_array_free(beta1_arr);
                }
                if !one_minus_beta1.is_null() {
                    mlx_array_free(one_minus_beta1);
                }
                if !beta2_arr.is_null() {
                    mlx_array_free(beta2_arr);
                }
                if !one_minus_beta2.is_null() {
                    mlx_array_free(one_minus_beta2);
                }
                if !inv_bc1.is_null() {
                    mlx_array_free(inv_bc1);
                }
                if !inv_bc2.is_null() {
                    mlx_array_free(inv_bc2);
                }
                if !eps_arr.is_null() {
                    mlx_array_free(eps_arr);
                }
                if !lr_arr.is_null() {
                    mlx_array_free(lr_arr);
                }
                if !decay_arr.is_null() {
                    mlx_array_free(decay_arr);
                }
                return Err(AutogradError::TapeInvariant(
                    "mlx_array_new_float32 returned null (adamw scalars)",
                ));
            }

            // m' = β1·m + (1-β1)·g
            let m_scaled = mlx_multiply(beta1_arr, m_handle.as_ptr());
            let g_scaled = mlx_multiply(one_minus_beta1, grad_arr);
            mlx_array_free(beta1_arr);
            mlx_array_free(one_minus_beta1);
            if m_scaled.is_null() || g_scaled.is_null() {
                if !m_scaled.is_null() {
                    mlx_array_free(m_scaled);
                }
                if !g_scaled.is_null() {
                    mlx_array_free(g_scaled);
                }
                mlx_array_free(grad_arr);
                mlx_array_free(beta2_arr);
                mlx_array_free(one_minus_beta2);
                mlx_array_free(inv_bc1);
                mlx_array_free(inv_bc2);
                mlx_array_free(eps_arr);
                mlx_array_free(lr_arr);
                mlx_array_free(decay_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_multiply returned null (adamw m update components)",
                ));
            }
            let new_m = mlx_add(m_scaled, g_scaled);
            mlx_array_free(m_scaled);
            mlx_array_free(g_scaled);
            if new_m.is_null() {
                mlx_array_free(grad_arr);
                mlx_array_free(beta2_arr);
                mlx_array_free(one_minus_beta2);
                mlx_array_free(inv_bc1);
                mlx_array_free(inv_bc2);
                mlx_array_free(eps_arr);
                mlx_array_free(lr_arr);
                mlx_array_free(decay_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_add returned null (adamw new_m)",
                ));
            }

            // v' = β2·v + (1-β2)·g²
            let g_sq = mlx_multiply(grad_arr, grad_arr);
            mlx_array_free(grad_arr);
            if g_sq.is_null() {
                mlx_array_free(new_m);
                mlx_array_free(beta2_arr);
                mlx_array_free(one_minus_beta2);
                mlx_array_free(inv_bc1);
                mlx_array_free(inv_bc2);
                mlx_array_free(eps_arr);
                mlx_array_free(lr_arr);
                mlx_array_free(decay_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_multiply returned null (adamw g²)",
                ));
            }
            let v_scaled = mlx_multiply(beta2_arr, v_handle.as_ptr());
            let gsq_scaled = mlx_multiply(one_minus_beta2, g_sq);
            mlx_array_free(beta2_arr);
            mlx_array_free(one_minus_beta2);
            mlx_array_free(g_sq);
            if v_scaled.is_null() || gsq_scaled.is_null() {
                if !v_scaled.is_null() {
                    mlx_array_free(v_scaled);
                }
                if !gsq_scaled.is_null() {
                    mlx_array_free(gsq_scaled);
                }
                mlx_array_free(new_m);
                mlx_array_free(inv_bc1);
                mlx_array_free(inv_bc2);
                mlx_array_free(eps_arr);
                mlx_array_free(lr_arr);
                mlx_array_free(decay_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_multiply returned null (adamw v update components)",
                ));
            }
            let new_v = mlx_add(v_scaled, gsq_scaled);
            mlx_array_free(v_scaled);
            mlx_array_free(gsq_scaled);
            if new_v.is_null() {
                mlx_array_free(new_m);
                mlx_array_free(inv_bc1);
                mlx_array_free(inv_bc2);
                mlx_array_free(eps_arr);
                mlx_array_free(lr_arr);
                mlx_array_free(decay_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_add returned null (adamw new_v)",
                ));
            }

            // m_hat = new_m / bc1 (multiply by precomputed reciprocal).
            let m_hat = mlx_multiply(new_m, inv_bc1);
            mlx_array_free(inv_bc1);
            if m_hat.is_null() {
                mlx_array_free(new_m);
                mlx_array_free(new_v);
                mlx_array_free(inv_bc2);
                mlx_array_free(eps_arr);
                mlx_array_free(lr_arr);
                mlx_array_free(decay_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_multiply returned null (adamw m_hat)",
                ));
            }
            let v_over_bc2 = mlx_multiply(new_v, inv_bc2);
            mlx_array_free(inv_bc2);
            if v_over_bc2.is_null() {
                mlx_array_free(m_hat);
                mlx_array_free(new_m);
                mlx_array_free(new_v);
                mlx_array_free(eps_arr);
                mlx_array_free(lr_arr);
                mlx_array_free(decay_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_multiply returned null (adamw v/bc2)",
                ));
            }
            let v_hat_sqrt = mlx_sqrt(v_over_bc2);
            mlx_array_free(v_over_bc2);
            if v_hat_sqrt.is_null() {
                mlx_array_free(m_hat);
                mlx_array_free(new_m);
                mlx_array_free(new_v);
                mlx_array_free(eps_arr);
                mlx_array_free(lr_arr);
                mlx_array_free(decay_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_sqrt returned null (adamw √(v/bc2))",
                ));
            }
            let denom = mlx_add(v_hat_sqrt, eps_arr);
            mlx_array_free(v_hat_sqrt);
            mlx_array_free(eps_arr);
            if denom.is_null() {
                mlx_array_free(m_hat);
                mlx_array_free(new_m);
                mlx_array_free(new_v);
                mlx_array_free(lr_arr);
                mlx_array_free(decay_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_add returned null (adamw denom)",
                ));
            }

            // update = lr · m_hat / denom = lr · m_hat · reciprocal(denom)
            let denom_recip = mlx_reciprocal(denom);
            mlx_array_free(denom);
            if denom_recip.is_null() {
                mlx_array_free(m_hat);
                mlx_array_free(new_m);
                mlx_array_free(new_v);
                mlx_array_free(lr_arr);
                mlx_array_free(decay_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_reciprocal returned null (adamw 1/denom)",
                ));
            }
            let ratio = mlx_multiply(m_hat, denom_recip);
            mlx_array_free(m_hat);
            mlx_array_free(denom_recip);
            if ratio.is_null() {
                mlx_array_free(new_m);
                mlx_array_free(new_v);
                mlx_array_free(lr_arr);
                mlx_array_free(decay_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_multiply returned null (adamw m_hat/denom)",
                ));
            }
            let scaled_update = mlx_multiply(lr_arr, ratio);
            mlx_array_free(lr_arr);
            mlx_array_free(ratio);
            if scaled_update.is_null() {
                mlx_array_free(new_m);
                mlx_array_free(new_v);
                mlx_array_free(decay_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_multiply returned null (adamw lr·ratio)",
                ));
            }

            // decayed_param = (1 - lr·wd) · param
            let decayed_param = mlx_multiply(decay_arr, param_handle.as_ptr());
            mlx_array_free(decay_arr);
            if decayed_param.is_null() {
                mlx_array_free(scaled_update);
                mlx_array_free(new_m);
                mlx_array_free(new_v);
                return Err(AutogradError::TapeInvariant(
                    "mlx_multiply returned null (adamw decay·param)",
                ));
            }

            // new_param = decayed_param - scaled_update
            let new_param = mlx_subtract(decayed_param, scaled_update);
            mlx_array_free(decayed_param);
            mlx_array_free(scaled_update);
            if new_param.is_null() {
                mlx_array_free(new_m);
                mlx_array_free(new_v);
                return Err(AutogradError::TapeInvariant(
                    "mlx_subtract returned null (adamw new_param)",
                ));
            }

            // M5.3b.11: no intra-op eval. The three MLX graph nodes return
            // unevaluated; `AdamW::step_device` collects every param's
            // triple and fires a single `backend.eval(...)` at the end of
            // the optimizer step, turning the per-step eval count from
            // `num_params` into `1` regardless of how many parameters the
            // model has. Lazy-graph composition across params is safe — the
            // updates are independent, no sub-node is shared.
            Ok((
                DeviceHandle::Metal(MlxHandle::from_raw(new_param)),
                DeviceHandle::Metal(MlxHandle::from_raw(new_m)),
                DeviceHandle::Metal(MlxHandle::from_raw(new_v)),
            ))
        }
    }
}

// Compute matmul gradients on-device. `grad_a = grad_out @ B^T` and
// `grad_b = A^T @ grad_out`; the inner-most two axes of A/B are transposed
// via `mlx_transpose_axes` (a lazy view that MLX fuses into the GEMM), so
// no host-side transpose or extra upload is needed. The `need_grad_a`/
// `need_grad_b` gates let the caller skip whichever SGEMM is not needed.
//
// One MLX round-trip per requested gradient: upload A, B, grad_out once,
// issue two matmuls and one eval, then copy the results back. Mirrors the
// forward's lock/eval/readback discipline.
fn mlx_matmul_backward(
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
        return Err(AutogradError::ShapeMismatch {
            expected: expected_out,
            got: grad_out_shape.to_vec(),
        });
    }
    let a_size: usize = a_shape.iter().product();
    let b_size: usize = b_shape.iter().product();
    let g_size: usize = grad_out_shape.iter().product();
    if a.len() != a_size {
        return Err(AutogradError::DataLengthMismatch {
            len: a.len(),
            shape: a_shape.to_vec(),
            size: a_size,
        });
    }
    if b.len() != b_size {
        return Err(AutogradError::DataLengthMismatch {
            len: b.len(),
            shape: b_shape.to_vec(),
            size: b_size,
        });
    }
    if grad_out.len() != g_size {
        return Err(AutogradError::DataLengthMismatch {
            len: grad_out.len(),
            shape: grad_out_shape.to_vec(),
            size: g_size,
        });
    }

    if !need_grad_a && !need_grad_b {
        return Ok((Vec::new(), Vec::new()));
    }

    let rank = a_shape.len();
    if rank != 2 && rank != 3 {
        return Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2 or rank-3",
            got: rank,
        });
    }

    // Axes permutation that swaps the last two dims (rank 2 or 3). MLX's
    // transpose_axes takes the full permutation vector.
    let transpose_axes: Vec<i32> = if rank == 2 { vec![1, 0] } else { vec![0, 2, 1] };

    let a_shape_i32: Vec<i32> = a_shape.iter().map(|&d| d as i32).collect();
    let b_shape_i32: Vec<i32> = b_shape.iter().map(|&d| d as i32).collect();
    let g_shape_i32: Vec<i32> = grad_out_shape.iter().map(|&d| d as i32).collect();

    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: `a`, `b`, `grad_out` outlive the FFI calls (MLX copies host
    // slices into its own storage); every MLX array allocated below is freed
    // on every return path.
    unsafe {
        let g_arr = mlx_array_from_data(
            grad_out.as_ptr() as *const c_void,
            g_shape_i32.as_ptr(),
            g_shape_i32.len() as i32,
            MLX_FLOAT32,
        );
        if g_arr.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null (grad_out)",
            ));
        }

        // grad_a = grad_out @ B^T
        let grad_a_host = if need_grad_a {
            let b_arr = mlx_array_from_data(
                b.as_ptr() as *const c_void,
                b_shape_i32.as_ptr(),
                b_shape_i32.len() as i32,
                MLX_FLOAT32,
            );
            if b_arr.is_null() {
                mlx_array_free(g_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_array_from_data returned null (b)",
                ));
            }
            let b_t = mlx_transpose_axes(b_arr, transpose_axes.as_ptr(), transpose_axes.len());
            mlx_array_free(b_arr);
            if b_t.is_null() {
                mlx_array_free(g_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_transpose_axes returned null (b)",
                ));
            }
            let out = mlx_matmul(g_arr, b_t);
            mlx_array_free(b_t);
            if out.is_null() {
                mlx_array_free(g_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_matmul returned null (grad_a)",
                ));
            }
            let host = match eval_and_readback(out) {
                Ok(h) => h,
                Err(e) => {
                    mlx_array_free(out);
                    mlx_array_free(g_arr);
                    return Err(e);
                }
            };
            mlx_array_free(out);
            if host.len() != a_size {
                mlx_array_free(g_arr);
                return Err(AutogradError::ShapeMismatch {
                    expected: a_shape.to_vec(),
                    got: vec![host.len()],
                });
            }
            host
        } else {
            Vec::new()
        };

        // grad_b = A^T @ grad_out
        let grad_b_host = if need_grad_b {
            let a_arr = mlx_array_from_data(
                a.as_ptr() as *const c_void,
                a_shape_i32.as_ptr(),
                a_shape_i32.len() as i32,
                MLX_FLOAT32,
            );
            if a_arr.is_null() {
                mlx_array_free(g_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_array_from_data returned null (a)",
                ));
            }
            let a_t = mlx_transpose_axes(a_arr, transpose_axes.as_ptr(), transpose_axes.len());
            mlx_array_free(a_arr);
            if a_t.is_null() {
                mlx_array_free(g_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_transpose_axes returned null (a)",
                ));
            }
            let out = mlx_matmul(a_t, g_arr);
            mlx_array_free(a_t);
            if out.is_null() {
                mlx_array_free(g_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_matmul returned null (grad_b)",
                ));
            }
            let host = match eval_and_readback(out) {
                Ok(h) => h,
                Err(e) => {
                    mlx_array_free(out);
                    mlx_array_free(g_arr);
                    return Err(e);
                }
            };
            mlx_array_free(out);
            if host.len() != b_size {
                mlx_array_free(g_arr);
                return Err(AutogradError::ShapeMismatch {
                    expected: b_shape.to_vec(),
                    got: vec![host.len()],
                });
            }
            host
        } else {
            Vec::new()
        };

        mlx_array_free(g_arr);
        Ok((grad_a_host, grad_b_host))
    }
}

// Shared shape validation for the lazy softmax / log_softmax device-handle
// paths. Pre-M5.3b.2 the eager `mlx_softmax_like` also checked `x.len()`
// against `product(shape)`; for the lazy path we trust MLX's own shape
// on the input handle (the caller built it via `upload(shape)`), so we
// only guard against degenerate rank / last-dim.
fn validate_softmax_shape(shape: &[usize]) -> Result<()> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    Ok(())
}

#[derive(Copy, Clone)]
enum SoftmaxKind {
    Softmax,
    LogSoftmax,
}

// Upload host slice → call mlx_softmax_axis (or x - logsumexp for log form)
// on axis=-1 → eval → copy back. The intermediate MLX arrays are freed
// explicitly so the host slice is the only authoritative copy, matching
// the matmul_forward contract.
fn mlx_softmax_like(x: &[f32], shape: &[usize], kind: SoftmaxKind) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected: usize = shape.iter().product();
    if x.len() != expected {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![x.len()],
        });
    }

    let shape_i32: Vec<i32> = shape.iter().map(|&dim| dim as i32).collect();
    // MLX treats -1 as "last axis"; using the signed form avoids a shape-len
    // dependency for ranks other than 2/3.
    let axis = -1_i32;

    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: `x` and `shape_i32` stay alive through the FFI call; MLX copies
    // from the host slice into its own array storage; every allocated MLX
    // array is freed in the same scope (softmax/logsumexp/subtract produce
    // fresh MLX nodes that we own here).
    unsafe {
        let input = mlx_array_from_data(
            x.as_ptr() as *const c_void,
            shape_i32.as_ptr(),
            shape_i32.len() as i32,
            MLX_FLOAT32,
        );
        if input.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }

        let out_arr = match kind {
            SoftmaxKind::Softmax => mlx_softmax_axis(input, axis, true),
            SoftmaxKind::LogSoftmax => {
                let lse = mlx_logsumexp_axis(input, axis, true);
                if lse.is_null() {
                    mlx_array_free(input);
                    return Err(AutogradError::TapeInvariant(
                        "mlx_logsumexp_axis returned null",
                    ));
                }
                let diff = mlx_subtract(input, lse);
                mlx_array_free(lse);
                diff
            }
        };
        if out_arr.is_null() {
            mlx_array_free(input);
            return Err(AutogradError::TapeInvariant(
                "mlx softmax/log_softmax returned null",
            ));
        }

        let mut eval_handles = [out_arr];
        mlx_eval(eval_handles.as_mut_ptr(), eval_handles.len());
        bump_eval_count();

        let size = mlx_array_size(out_arr);
        let data_ptr = mlx_array_data_float32(out_arr);
        if data_ptr.is_null() {
            mlx_array_free(input);
            mlx_array_free(out_arr);
            return Err(AutogradError::TapeInvariant(
                "mlx_array_data_float32 returned null",
            ));
        }

        let mut host = vec![0.0f32; size];
        std::ptr::copy_nonoverlapping(data_ptr, host.as_mut_ptr(), size);

        mlx_array_free(input);
        mlx_array_free(out_arr);

        if host.len() != expected {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![expected],
                got: vec![host.len()],
            });
        }
        Ok(host)
    }
}

// === Helpers for the additive trait methods ===

#[derive(Copy, Clone)]
enum UnaryOp {
    Exp,
    Neg,
    Gelu,
    Silu,
    MulScalar(f32),
}

#[derive(Copy, Clone)]
enum BinaryOp {
    Mul,
}

#[derive(Copy, Clone)]
enum ReduceOp {
    Sum,
    Mean,
}

// Upload a 1-D host slice → apply a single MLX op producing a same-sized
// array → eval → copy back → free. All MLX calls run under `MLX_GUARD`.
fn mlx_unary_flat(a: &[f32], op: UnaryOp) -> Result<Vec<f32>> {
    let n = a.len();
    let shape_i32 = [n as i32];
    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: `a` outlives the FFI call (MLX copies from the host slice);
    // every MLX array allocated here is freed on every path before return,
    // and `MLX_GUARD` serializes all MLX state.
    unsafe {
        let input = mlx_array_from_data(
            a.as_ptr() as *const c_void,
            shape_i32.as_ptr(),
            1,
            MLX_FLOAT32,
        );
        if input.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }

        let out_arr = match op {
            UnaryOp::Exp => mlx_exp(input),
            UnaryOp::Neg => mlx_negative(input),
            UnaryOp::MulScalar(s) => {
                let scalar = mlx_array_new_float32(s);
                if scalar.is_null() {
                    mlx_array_free(input);
                    return Err(AutogradError::TapeInvariant(
                        "mlx_array_new_float32 returned null",
                    ));
                }
                let out = mlx_multiply(input, scalar);
                mlx_array_free(scalar);
                out
            }
            UnaryOp::Gelu => gelu_tanh(input)?,
            UnaryOp::Silu => {
                let sig = mlx_sigmoid(input);
                if sig.is_null() {
                    mlx_array_free(input);
                    return Err(AutogradError::TapeInvariant("mlx_sigmoid returned null"));
                }
                let out = mlx_multiply(input, sig);
                mlx_array_free(sig);
                out
            }
        };
        if out_arr.is_null() {
            mlx_array_free(input);
            return Err(AutogradError::TapeInvariant("mlx unary op returned null"));
        }

        let host = eval_and_readback(out_arr)?;
        mlx_array_free(input);
        mlx_array_free(out_arr);

        if host.len() != n {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![n],
                got: vec![host.len()],
            });
        }
        Ok(host)
    }
}

// Compose `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` using MLX
// primitives. Matches `cpu_gelu_forward` (tanh approximation) within f32
// precision. `input` is borrowed; the returned array is freshly owned.
//
// Safety: caller holds `MLX_GUARD`; `input` is a live MLX array; every
// intermediate we allocate here is freed before returning.
unsafe fn gelu_tanh(input: *mut mlx_sys::mlx_array) -> Result<*mut mlx_sys::mlx_array> {
    const K: f32 = 0.797_884_6_f32; // sqrt(2/pi)
    unsafe {
        // xsq = x * x ; xcube = xsq * x
        let xsq = mlx_multiply(input, input);
        if xsq.is_null() {
            return Err(AutogradError::TapeInvariant("gelu: xsq null"));
        }
        let xcube = mlx_multiply(xsq, input);
        mlx_array_free(xsq);
        if xcube.is_null() {
            return Err(AutogradError::TapeInvariant("gelu: xcube null"));
        }

        // inner = K * (x + 0.044715 * xcube)
        let coef = mlx_array_new_float32(0.044_715_f32);
        let coef_times_cube = mlx_multiply(coef, xcube);
        mlx_array_free(coef);
        mlx_array_free(xcube);
        if coef_times_cube.is_null() {
            return Err(AutogradError::TapeInvariant("gelu: coef*cube null"));
        }
        let inner_sum = mlx_add(input, coef_times_cube);
        mlx_array_free(coef_times_cube);
        if inner_sum.is_null() {
            return Err(AutogradError::TapeInvariant("gelu: inner_sum null"));
        }
        let k_scalar = mlx_array_new_float32(K);
        let inner = mlx_multiply(k_scalar, inner_sum);
        mlx_array_free(k_scalar);
        mlx_array_free(inner_sum);
        if inner.is_null() {
            return Err(AutogradError::TapeInvariant("gelu: inner null"));
        }

        // tanh_val = tanh(inner); one_plus = 1 + tanh_val
        let tanh_val = mlx_tanh(inner);
        mlx_array_free(inner);
        if tanh_val.is_null() {
            return Err(AutogradError::TapeInvariant("gelu: tanh null"));
        }
        let one = mlx_array_new_float32(1.0_f32);
        let one_plus = mlx_add(one, tanh_val);
        mlx_array_free(one);
        mlx_array_free(tanh_val);
        if one_plus.is_null() {
            return Err(AutogradError::TapeInvariant("gelu: 1+tanh null"));
        }

        // out = 0.5 * x * one_plus
        let half = mlx_array_new_float32(0.5_f32);
        let half_x = mlx_multiply(half, input);
        mlx_array_free(half);
        if half_x.is_null() {
            mlx_array_free(one_plus);
            return Err(AutogradError::TapeInvariant("gelu: half*x null"));
        }
        let out = mlx_multiply(half_x, one_plus);
        mlx_array_free(half_x);
        mlx_array_free(one_plus);
        if out.is_null() {
            return Err(AutogradError::TapeInvariant("gelu: final null"));
        }
        Ok(out)
    }
}

fn mlx_binary_flat(a: &[f32], b: &[f32], op: BinaryOp) -> Result<Vec<f32>> {
    let n = a.len();
    let shape_i32 = [n as i32];
    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: `a`/`b` outlive the FFI call (MLX copies host slices); both
    // allocated inputs plus the result are freed before return.
    unsafe {
        let a_arr = mlx_array_from_data(
            a.as_ptr() as *const c_void,
            shape_i32.as_ptr(),
            1,
            MLX_FLOAT32,
        );
        if a_arr.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }
        let b_arr = mlx_array_from_data(
            b.as_ptr() as *const c_void,
            shape_i32.as_ptr(),
            1,
            MLX_FLOAT32,
        );
        if b_arr.is_null() {
            mlx_array_free(a_arr);
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }
        let out_arr = match op {
            BinaryOp::Mul => mlx_multiply(a_arr, b_arr),
        };
        if out_arr.is_null() {
            mlx_array_free(a_arr);
            mlx_array_free(b_arr);
            return Err(AutogradError::TapeInvariant("mlx binary op returned null"));
        }
        let host = eval_and_readback(out_arr)?;
        mlx_array_free(a_arr);
        mlx_array_free(b_arr);
        mlx_array_free(out_arr);

        if host.len() != n {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![n],
                got: vec![host.len()],
            });
        }
        Ok(host)
    }
}

// Right-aligned broadcast-add via MLX's native NumPy-style broadcasting:
// `mlx_add(a, b)` accepts operands with different but right-broadcast-compatible
// shapes and returns an array with the broadcast shape — which, for our
// contract (b_shape.len() <= a_shape.len() and each b-axis is 1 or matches),
// equals `a_shape`. No explicit reshape is required on the host side.
fn mlx_add_broadcast(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<Vec<f32>> {
    validate_broadcast(a_shape, b_shape)?;
    let a_size: usize = if a_shape.is_empty() {
        1
    } else {
        a_shape.iter().product()
    };
    let b_size: usize = if b_shape.is_empty() {
        1
    } else {
        b_shape.iter().product()
    };
    if a.len() != a_size {
        return Err(AutogradError::DataLengthMismatch {
            len: a.len(),
            shape: a_shape.to_vec(),
            size: a_size,
        });
    }
    if b.len() != b_size {
        return Err(AutogradError::DataLengthMismatch {
            len: b.len(),
            shape: b_shape.to_vec(),
            size: b_size,
        });
    }

    let a_shape_i32: Vec<i32> = a_shape.iter().map(|&d| d as i32).collect();
    let b_shape_i32: Vec<i32> = b_shape.iter().map(|&d| d as i32).collect();
    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: host slices `a`/`b` outlive the FFI call (MLX copies from
    // them). Every MLX array we allocate is freed on every return path.
    unsafe {
        let a_arr = mlx_array_from_data(
            a.as_ptr() as *const c_void,
            a_shape_i32.as_ptr(),
            a_shape_i32.len() as i32,
            MLX_FLOAT32,
        );
        if a_arr.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }
        let b_arr = mlx_array_from_data(
            b.as_ptr() as *const c_void,
            b_shape_i32.as_ptr(),
            b_shape_i32.len() as i32,
            MLX_FLOAT32,
        );
        if b_arr.is_null() {
            mlx_array_free(a_arr);
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }
        let out_arr = mlx_add(a_arr, b_arr);
        if out_arr.is_null() {
            mlx_array_free(a_arr);
            mlx_array_free(b_arr);
            return Err(AutogradError::TapeInvariant(
                "mlx_add returned null (broadcast)",
            ));
        }
        let host = eval_and_readback(out_arr)?;
        mlx_array_free(a_arr);
        mlx_array_free(b_arr);
        mlx_array_free(out_arr);

        if host.len() != a_size {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![a_size],
                got: vec![host.len()],
            });
        }
        Ok(host)
    }
}

// Lazy sibling of `mlx_rms_norm`: borrows an existing MLX `x` handle,
// uploads `weight` as a temporary array, calls `mlx_fast_rms_norm`, and
// returns the result wrapped in an `MlxHandle` without calling
// `mlx_eval`. Mirrors the shape validation from `mlx_rms_norm`. M5.3b.6.
fn mlx_rms_norm_lazy(
    x_ptr: *mut mlx_array,
    weight: &[f32],
    shape: &[usize],
    eps: f32,
) -> Result<DeviceHandle> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    if weight.len() != last_dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![last_dim],
            got: vec![weight.len()],
        });
    }

    let w_shape = [last_dim as i32];
    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: `x_ptr` is borrowed for the duration of this call; `w_arr`
    // is allocated here and freed before return; the returned `MlxHandle`
    // owns the final `mlx_fast_rms_norm` result.
    unsafe {
        let w_arr = mlx_array_from_data(
            weight.as_ptr() as *const c_void,
            w_shape.as_ptr(),
            1,
            MLX_FLOAT32,
        );
        if w_arr.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null (rms_norm lazy weight)",
            ));
        }
        let out_arr = mlx_fast_rms_norm(x_ptr, w_arr, eps);
        mlx_array_free(w_arr);
        if out_arr.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_fast_rms_norm returned null (lazy)",
            ));
        }
        Ok(DeviceHandle::Metal(MlxHandle::from_raw(out_arr)))
    }
}

fn mlx_gelu_erf_lazy(x_ptr: *mut mlx_array) -> Result<DeviceHandle> {
    // GELU_erf(x) = 0.5 * x * (1 + erf(x * INV_SQRT_2))
    // Composed lazily so it merges with upstream matmul + downstream
    // rmsnorm in MLX's lazy graph. Formula mirrors `ops::activation::gelu`
    // CPU body exactly (same constant, same order of operations); the
    // chosen `mlx_erf` matches `libm::erff` to within the ULP range MLX
    // uses for f32 erf — parity test gates at 1e-4.
    const INV_SQRT_2: f32 = 0.707_106_77_f32;
    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: `x_ptr` is borrowed for the duration of this call. Every
    // intermediate we allocate is freed before the fn returns. The final
    // returned handle owns the last `mlx_multiply` result.
    unsafe {
        let k_arr = mlx_array_new_float32(INV_SQRT_2);
        if k_arr.is_null() {
            return Err(AutogradError::TapeInvariant("gelu lazy: k scalar null"));
        }
        let scaled = mlx_multiply(k_arr, x_ptr);
        mlx_array_free(k_arr);
        if scaled.is_null() {
            return Err(AutogradError::TapeInvariant("gelu lazy: scaled null"));
        }
        let erf_val = mlx_erf(scaled);
        mlx_array_free(scaled);
        if erf_val.is_null() {
            return Err(AutogradError::TapeInvariant("gelu lazy: erf null"));
        }
        let one = mlx_array_new_float32(1.0_f32);
        if one.is_null() {
            mlx_array_free(erf_val);
            return Err(AutogradError::TapeInvariant("gelu lazy: one null"));
        }
        let one_plus = mlx_add(one, erf_val);
        mlx_array_free(one);
        mlx_array_free(erf_val);
        if one_plus.is_null() {
            return Err(AutogradError::TapeInvariant("gelu lazy: 1+erf null"));
        }
        let half = mlx_array_new_float32(0.5_f32);
        if half.is_null() {
            mlx_array_free(one_plus);
            return Err(AutogradError::TapeInvariant("gelu lazy: half null"));
        }
        let half_x = mlx_multiply(half, x_ptr);
        mlx_array_free(half);
        if half_x.is_null() {
            mlx_array_free(one_plus);
            return Err(AutogradError::TapeInvariant("gelu lazy: half*x null"));
        }
        let out = mlx_multiply(half_x, one_plus);
        mlx_array_free(half_x);
        mlx_array_free(one_plus);
        if out.is_null() {
            return Err(AutogradError::TapeInvariant("gelu lazy: final null"));
        }
        Ok(DeviceHandle::Metal(MlxHandle::from_raw(out)))
    }
}

fn mlx_embedding_lazy(
    table_ptr: *mut mlx_array,
    table_shape: &[usize],
    ids: &[i32],
) -> Result<DeviceHandle> {
    if table_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "2",
            got: table_shape.len(),
        });
    }
    let vocab = table_shape[0];
    let hidden = table_shape[1];
    let seq = ids.len();
    for &id in ids {
        if id < 0 || (id as usize) >= vocab {
            return Err(AutogradError::IndexOutOfBounds {
                index: id as usize,
                upper: vocab,
            });
        }
    }

    let ids_shape = [seq as i32];
    let reshape_shape = [1i32, seq as i32, hidden as i32];
    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: `table_ptr` is borrowed for the duration of this call. The
    // int32 ids array is allocated here and freed before return. The
    // intermediate `gathered` is freed after reshape; the returned handle
    // owns the reshape output.
    unsafe {
        let ids_arr = mlx_array_from_data(
            ids.as_ptr() as *const c_void,
            ids_shape.as_ptr(),
            1,
            MLX_INT32,
        );
        if ids_arr.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null (embedding lazy ids)",
            ));
        }
        let gathered = mlx_take_axis(table_ptr, ids_arr, 0);
        mlx_array_free(ids_arr);
        if gathered.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_take_axis returned null (embedding lazy)",
            ));
        }
        let reshaped = mlx_reshape(gathered, reshape_shape.as_ptr(), 3);
        mlx_array_free(gathered);
        if reshaped.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_reshape returned null (embedding lazy)",
            ));
        }
        Ok(DeviceHandle::Metal(MlxHandle::from_raw(reshaped)))
    }
}

fn mlx_gather_last_dim_lazy(
    src_ptr: *mut mlx_array,
    src_shape: &[usize],
    ids: &[i32],
) -> Result<DeviceHandle> {
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
    if ids.len() != prefix {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix,
            got: ids.len(),
        });
    }
    for &id in ids {
        if id < 0 || (id as usize) >= vocab {
            return Err(AutogradError::IndexOutOfBounds {
                index: id as usize,
                upper: vocab,
            });
        }
    }

    // Remap to flat ids on the host: one int32 per output row. Tiny
    // buffer, re-uploaded per call — the prefix product is typically
    // `B * S` which is dwarfed by the flattened src.
    let vocab_i32 = vocab as i32;
    let flat_ids: Vec<i32> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (i as i32) * vocab_i32 + id)
        .collect();

    let flat_src_shape = [(prefix * vocab) as i32];
    let ids_shape = [prefix as i32];
    let out_shape_i32: Vec<i32> = src_shape[..src_shape.len() - 1]
        .iter()
        .map(|&d| d as i32)
        .collect();
    let out_ndim = out_shape_i32.len();

    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: `src_ptr` is borrowed for the duration of this call. The
    // int32 ids array and intermediate `flat` / `gathered` views are
    // allocated here and freed before return; the returned handle owns
    // the final reshape output.
    unsafe {
        let flat = mlx_reshape(src_ptr, flat_src_shape.as_ptr(), 1);
        if flat.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_reshape returned null (gather lazy flatten)",
            ));
        }
        let ids_arr = mlx_array_from_data(
            flat_ids.as_ptr() as *const c_void,
            ids_shape.as_ptr(),
            1,
            MLX_INT32,
        );
        if ids_arr.is_null() {
            mlx_array_free(flat);
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null (gather lazy ids)",
            ));
        }
        let gathered = mlx_take_axis(flat, ids_arr, 0);
        mlx_array_free(flat);
        mlx_array_free(ids_arr);
        if gathered.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_take_axis returned null (gather lazy)",
            ));
        }
        let reshaped = mlx_reshape(gathered, out_shape_i32.as_ptr(), out_ndim);
        mlx_array_free(gathered);
        if reshaped.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_reshape returned null (gather lazy out)",
            ));
        }
        Ok(DeviceHandle::Metal(MlxHandle::from_raw(reshaped)))
    }
}

fn mlx_rms_norm(x: &[f32], weight: &[f32], shape: &[usize], eps: f32) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected: usize = shape.iter().product();
    if x.len() != expected {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![x.len()],
        });
    }
    if weight.len() != last_dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![last_dim],
            got: vec![weight.len()],
        });
    }

    let shape_i32: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    let w_shape = [last_dim as i32];
    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: both host slices live across the FFI call (MLX copies), and
    // every MLX array allocated here is freed before return.
    unsafe {
        let x_arr = mlx_array_from_data(
            x.as_ptr() as *const c_void,
            shape_i32.as_ptr(),
            shape_i32.len() as i32,
            MLX_FLOAT32,
        );
        if x_arr.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }
        let w_arr = mlx_array_from_data(
            weight.as_ptr() as *const c_void,
            w_shape.as_ptr(),
            1,
            MLX_FLOAT32,
        );
        if w_arr.is_null() {
            mlx_array_free(x_arr);
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }
        let out_arr = mlx_fast_rms_norm(x_arr, w_arr, eps);
        if out_arr.is_null() {
            mlx_array_free(x_arr);
            mlx_array_free(w_arr);
            return Err(AutogradError::TapeInvariant(
                "mlx_fast_rms_norm returned null",
            ));
        }
        let host = eval_and_readback(out_arr)?;
        mlx_array_free(x_arr);
        mlx_array_free(w_arr);
        mlx_array_free(out_arr);

        if host.len() != expected {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![expected],
                got: vec![host.len()],
            });
        }
        Ok(host)
    }
}

fn mlx_embedding(weight: &[f32], vocab: usize, dim: usize, ids: &[i32]) -> Result<Vec<f32>> {
    if weight.len() != vocab * dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![vocab * dim],
            got: vec![weight.len()],
        });
    }
    let n_ids = ids.len();
    let out_elems = n_ids * dim;
    if n_ids == 0 {
        return Ok(Vec::new());
    }

    // Sanitize ids on the host: clamp OOB / negative to 0 so `mlx_take_axis`
    // never trips a bounds assertion, and track which output rows must be
    // zeroed (matches `cpu_embedding_forward` behavior).
    let mut safe_ids: Vec<i32> = Vec::with_capacity(n_ids);
    let mut row_mask: Vec<f32> = Vec::with_capacity(n_ids);
    let mut has_invalid = false;
    for &id in ids {
        if id < 0 || (id as usize) >= vocab {
            safe_ids.push(0);
            row_mask.push(0.0);
            has_invalid = true;
        } else {
            safe_ids.push(id);
            row_mask.push(1.0);
        }
    }

    let weight_shape = [vocab as i32, dim as i32];
    let ids_shape = [n_ids as i32];
    // mask is `[n_ids, 1]` so it broadcasts across the `dim` axis.
    let mask_shape = [n_ids as i32, 1];
    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: `weight`, `safe_ids`, `row_mask` all outlive the FFI calls
    // below (MLX copies host slices into its own storage); every array we
    // allocate is freed before return.
    unsafe {
        let w_arr = mlx_array_from_data(
            weight.as_ptr() as *const c_void,
            weight_shape.as_ptr(),
            2,
            MLX_FLOAT32,
        );
        if w_arr.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }
        let ids_arr = mlx_array_from_data(
            safe_ids.as_ptr() as *const c_void,
            ids_shape.as_ptr(),
            1,
            MLX_INT32,
        );
        if ids_arr.is_null() {
            mlx_array_free(w_arr);
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }
        let gathered = mlx_take_axis(w_arr, ids_arr, 0);
        mlx_array_free(w_arr);
        mlx_array_free(ids_arr);
        if gathered.is_null() {
            return Err(AutogradError::TapeInvariant("mlx_take_axis returned null"));
        }

        let out_arr = if has_invalid {
            let mask_arr = mlx_array_from_data(
                row_mask.as_ptr() as *const c_void,
                mask_shape.as_ptr(),
                2,
                MLX_FLOAT32,
            );
            if mask_arr.is_null() {
                mlx_array_free(gathered);
                return Err(AutogradError::TapeInvariant(
                    "mlx_array_from_data returned null",
                ));
            }
            let masked = mlx_multiply(gathered, mask_arr);
            mlx_array_free(mask_arr);
            mlx_array_free(gathered);
            if masked.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_multiply returned null (embedding mask)",
                ));
            }
            masked
        } else {
            gathered
        };

        let host = eval_and_readback(out_arr)?;
        mlx_array_free(out_arr);

        if host.len() != out_elems {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![out_elems],
                got: vec![host.len()],
            });
        }
        Ok(host)
    }
}

fn mlx_reduce_last_axis(x: &[f32], shape: &[usize], op: ReduceOp) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected: usize = shape.iter().product();
    if x.len() != expected {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![x.len()],
        });
    }
    let out_elems = expected / last_dim;
    let shape_i32: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: `x` outlives the FFI call (MLX copies); both the input and
    // reduced arrays are freed on every return path.
    unsafe {
        let input = mlx_array_from_data(
            x.as_ptr() as *const c_void,
            shape_i32.as_ptr(),
            shape_i32.len() as i32,
            MLX_FLOAT32,
        );
        if input.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }
        let out_arr = match op {
            ReduceOp::Sum => mlx_sum_axis(input, -1, false),
            ReduceOp::Mean => mlx_mean_axis(input, -1, false),
        };
        if out_arr.is_null() {
            mlx_array_free(input);
            return Err(AutogradError::TapeInvariant("mlx reduce returned null"));
        }
        let host = eval_and_readback(out_arr)?;
        mlx_array_free(input);
        mlx_array_free(out_arr);

        if host.len() != out_elems {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![out_elems],
                got: vec![host.len()],
            });
        }
        Ok(host)
    }
}

fn mlx_rope(x: &[f32], x_shape: &[usize], cos: &[f32], sin: &[f32]) -> Result<Vec<f32>> {
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

    let x_shape_i32: [i32; 4] = [batch as i32, heads as i32, seq as i32, head_dim as i32];
    // cos/sin are uploaded as `[1, 1, seq, half_dim]` so they broadcast over
    // [B, H] during the multiplies without allocating a full expanded tensor.
    let cache_shape_i32: [i32; 4] = [1, 1, seq as i32, half_dim as i32];
    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: all three host slices live across the FFI calls (MLX copies);
    // every MLX array allocated below is freed before any early return.
    unsafe {
        let x_arr = mlx_array_from_data(
            x.as_ptr() as *const c_void,
            x_shape_i32.as_ptr(),
            4,
            MLX_FLOAT32,
        );
        if x_arr.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }
        let cos_arr = mlx_array_from_data(
            cos.as_ptr() as *const c_void,
            cache_shape_i32.as_ptr(),
            4,
            MLX_FLOAT32,
        );
        if cos_arr.is_null() {
            mlx_array_free(x_arr);
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }
        let sin_arr = mlx_array_from_data(
            sin.as_ptr() as *const c_void,
            cache_shape_i32.as_ptr(),
            4,
            MLX_FLOAT32,
        );
        if sin_arr.is_null() {
            mlx_array_free(x_arr);
            mlx_array_free(cos_arr);
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }

        // x0 = x[..., :half_dim] ; x1 = x[..., half_dim:]
        // mlx_slice takes start/stop/strides per dim. ndim=4, strides=all 1.
        let starts_lo: [i32; 4] = [0, 0, 0, 0];
        let stops_lo: [i32; 4] = [batch as i32, heads as i32, seq as i32, half_dim as i32];
        let starts_hi: [i32; 4] = [0, 0, 0, half_dim as i32];
        let stops_hi: [i32; 4] = [batch as i32, heads as i32, seq as i32, head_dim as i32];
        let strides: [i32; 4] = [1, 1, 1, 1];

        let x0 = mlx_slice(
            x_arr,
            starts_lo.as_ptr(),
            stops_lo.as_ptr(),
            strides.as_ptr(),
            4,
        );
        let x1 = mlx_slice(
            x_arr,
            starts_hi.as_ptr(),
            stops_hi.as_ptr(),
            strides.as_ptr(),
            4,
        );
        mlx_array_free(x_arr);
        if x0.is_null() || x1.is_null() {
            if !x0.is_null() {
                mlx_array_free(x0);
            }
            if !x1.is_null() {
                mlx_array_free(x1);
            }
            mlx_array_free(cos_arr);
            mlx_array_free(sin_arr);
            return Err(AutogradError::TapeInvariant("mlx_slice returned null"));
        }

        // out0 = x0 * cos - x1 * sin
        // out1 = x1 * cos + x0 * sin
        let x0c = mlx_multiply(x0, cos_arr);
        let x1s = mlx_multiply(x1, sin_arr);
        let x1c = mlx_multiply(x1, cos_arr);
        let x0s = mlx_multiply(x0, sin_arr);
        mlx_array_free(x0);
        mlx_array_free(x1);
        mlx_array_free(cos_arr);
        mlx_array_free(sin_arr);
        if x0c.is_null() || x1s.is_null() || x1c.is_null() || x0s.is_null() {
            for p in [x0c, x1s, x1c, x0s] {
                if !p.is_null() {
                    mlx_array_free(p);
                }
            }
            return Err(AutogradError::TapeInvariant("mlx rope multiply null"));
        }
        let out0 = mlx_subtract(x0c, x1s);
        let out1 = mlx_add(x1c, x0s);
        mlx_array_free(x0c);
        mlx_array_free(x1s);
        mlx_array_free(x1c);
        mlx_array_free(x0s);
        if out0.is_null() || out1.is_null() {
            if !out0.is_null() {
                mlx_array_free(out0);
            }
            if !out1.is_null() {
                mlx_array_free(out1);
            }
            return Err(AutogradError::TapeInvariant("mlx rope add/subtract null"));
        }

        // Concatenate along the last axis (axis=3 for rank-4).
        let mut parts = [out0, out1];
        let concat = mlx_concatenate_axis(parts.as_mut_ptr(), 2, 3);
        mlx_array_free(out0);
        mlx_array_free(out1);
        if concat.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_concatenate_axis returned null",
            ));
        }

        let host = eval_and_readback(concat)?;
        mlx_array_free(concat);

        if host.len() != expected_x {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![expected_x],
                got: vec![host.len()],
            });
        }
        Ok(host)
    }
}

// Lazy sibling of `mlx_rope`: borrows an existing MLX `x` handle, uploads
// cos/sin as temporary arrays, composes the same rotation graph, and
// returns the final concat node wrapped in an `MlxHandle` without calling
// `mlx_eval`. Mirrors every shape/null check in `mlx_rope` but is scoped
// to the device-handle path. M5.3b.5.
fn mlx_rope_lazy(
    x_ptr: *mut mlx_array,
    x_shape: &[usize],
    cos: &[f32],
    sin: &[f32],
) -> Result<DeviceHandle> {
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
    let expected_cache = seq * half_dim;
    if cos.len() != expected_cache || sin.len() != expected_cache {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected_cache],
            got: vec![cos.len().min(sin.len())],
        });
    }

    let cache_shape_i32: [i32; 4] = [1, 1, seq as i32, half_dim as i32];
    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: `x_ptr` is borrowed for the duration of this call; every
    // MLX array allocated below is freed before any early return; the
    // returned `MlxHandle` owns the final concat result.
    unsafe {
        let cos_arr = mlx_array_from_data(
            cos.as_ptr() as *const c_void,
            cache_shape_i32.as_ptr(),
            4,
            MLX_FLOAT32,
        );
        if cos_arr.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null (rope lazy cos)",
            ));
        }
        let sin_arr = mlx_array_from_data(
            sin.as_ptr() as *const c_void,
            cache_shape_i32.as_ptr(),
            4,
            MLX_FLOAT32,
        );
        if sin_arr.is_null() {
            mlx_array_free(cos_arr);
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null (rope lazy sin)",
            ));
        }

        let starts_lo: [i32; 4] = [0, 0, 0, 0];
        let stops_lo: [i32; 4] = [batch as i32, heads as i32, seq as i32, half_dim as i32];
        let starts_hi: [i32; 4] = [0, 0, 0, half_dim as i32];
        let stops_hi: [i32; 4] = [batch as i32, heads as i32, seq as i32, head_dim as i32];
        let strides: [i32; 4] = [1, 1, 1, 1];

        let x0 = mlx_slice(
            x_ptr,
            starts_lo.as_ptr(),
            stops_lo.as_ptr(),
            strides.as_ptr(),
            4,
        );
        let x1 = mlx_slice(
            x_ptr,
            starts_hi.as_ptr(),
            stops_hi.as_ptr(),
            strides.as_ptr(),
            4,
        );
        if x0.is_null() || x1.is_null() {
            if !x0.is_null() {
                mlx_array_free(x0);
            }
            if !x1.is_null() {
                mlx_array_free(x1);
            }
            mlx_array_free(cos_arr);
            mlx_array_free(sin_arr);
            return Err(AutogradError::TapeInvariant(
                "mlx_slice returned null (rope lazy)",
            ));
        }

        let x0c = mlx_multiply(x0, cos_arr);
        let x1s = mlx_multiply(x1, sin_arr);
        let x1c = mlx_multiply(x1, cos_arr);
        let x0s = mlx_multiply(x0, sin_arr);
        mlx_array_free(x0);
        mlx_array_free(x1);
        mlx_array_free(cos_arr);
        mlx_array_free(sin_arr);
        if x0c.is_null() || x1s.is_null() || x1c.is_null() || x0s.is_null() {
            for p in [x0c, x1s, x1c, x0s] {
                if !p.is_null() {
                    mlx_array_free(p);
                }
            }
            return Err(AutogradError::TapeInvariant(
                "mlx rope multiply null (lazy)",
            ));
        }
        let out0 = mlx_subtract(x0c, x1s);
        let out1 = mlx_add(x1c, x0s);
        mlx_array_free(x0c);
        mlx_array_free(x1s);
        mlx_array_free(x1c);
        mlx_array_free(x0s);
        if out0.is_null() || out1.is_null() {
            if !out0.is_null() {
                mlx_array_free(out0);
            }
            if !out1.is_null() {
                mlx_array_free(out1);
            }
            return Err(AutogradError::TapeInvariant(
                "mlx rope add/subtract null (lazy)",
            ));
        }

        let mut parts = [out0, out1];
        let concat = mlx_concatenate_axis(parts.as_mut_ptr(), 2, 3);
        mlx_array_free(out0);
        mlx_array_free(out1);
        if concat.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_concatenate_axis returned null (rope lazy)",
            ));
        }

        Ok(DeviceHandle::Metal(MlxHandle::from_raw(concat)))
    }
}

fn mlx_gather_last_dim(src: &[f32], src_shape: &[usize], ids: &[i32]) -> Result<Vec<f32>> {
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
    for &id in ids {
        if id < 0 || (id as usize) >= vocab {
            return Err(AutogradError::IndexOutOfBounds {
                index: id as usize,
                upper: vocab,
            });
        }
    }

    // Flatten src to `[prefix * vocab]`, then take a single flat index per
    // output position (`i * vocab + ids[i]`). One `mlx_take_axis` call gives
    // the whole result — no per-row loop.
    let vocab_i32 = vocab as i32;
    let flat_ids: Vec<i32> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (i as i32) * vocab_i32 + id)
        .collect();

    let src_flat_shape = [(prefix * vocab) as i32];
    let ids_shape = [prefix as i32];
    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: `src` and `flat_ids` both outlive the FFI calls; every MLX
    // array allocated here is freed before return.
    unsafe {
        let src_arr = mlx_array_from_data(
            src.as_ptr() as *const c_void,
            src_flat_shape.as_ptr(),
            1,
            MLX_FLOAT32,
        );
        if src_arr.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }
        let ids_arr = mlx_array_from_data(
            flat_ids.as_ptr() as *const c_void,
            ids_shape.as_ptr(),
            1,
            MLX_INT32,
        );
        if ids_arr.is_null() {
            mlx_array_free(src_arr);
            return Err(AutogradError::TapeInvariant(
                "mlx_array_from_data returned null",
            ));
        }
        let out_arr = mlx_take_axis(src_arr, ids_arr, 0);
        mlx_array_free(src_arr);
        mlx_array_free(ids_arr);
        if out_arr.is_null() {
            return Err(AutogradError::TapeInvariant("mlx_take_axis returned null"));
        }
        let host = eval_and_readback(out_arr)?;
        mlx_array_free(out_arr);

        if host.len() != prefix {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![prefix],
                got: vec![host.len()],
            });
        }
        Ok(host)
    }
}

// Scatter-add `prefix_rows` feature vectors into a zero-initialized
// `[vocab, feature_dim]` output buffer. Matches `cpu_scatter_add_rows_forward`
// semantics: negative or OOB indices are silently skipped; aliased indices
// accumulate via MLX's `scatter_add` (atomic/additive, not overwrite).
//
// OOB/negative filtering happens host-side here — the C++ helper assumes
// pre-sanitized in-range indices. This mirrors `mlx_embedding`'s approach,
// with the difference that we drop invalid rows entirely (no row_mask) since
// scatter_add would still fault on an OOB destination.
fn mlx_scatter_add_rows(
    upstream: &[f32],
    prefix_rows: usize,
    feature_dim: usize,
    indices: &[i32],
    vocab: usize,
) -> Result<Vec<f32>> {
    let expected_upstream = prefix_rows * feature_dim;
    if upstream.len() != expected_upstream {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected_upstream],
            got: vec![upstream.len()],
        });
    }
    if indices.len() != prefix_rows {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix_rows,
            got: indices.len(),
        });
    }
    let out_len = vocab * feature_dim;

    // Fast paths: empty output or empty work → return zeros without touching
    // MLX. Vocab==0 means the caller has nothing to accumulate into.
    if out_len == 0 {
        return Ok(Vec::new());
    }
    if prefix_rows == 0 || feature_dim == 0 {
        return Ok(vec![0.0_f32; out_len]);
    }

    // Filter OOB/negative indices host-side; collect compact (updates, indices)
    // pairs so the FFI path sees only in-range entries.
    let mut safe_indices: Vec<i32> = Vec::with_capacity(prefix_rows);
    let mut safe_updates: Vec<f32> = Vec::with_capacity(expected_upstream);
    for (row, &id) in indices.iter().enumerate() {
        if id < 0 || (id as usize) >= vocab {
            continue;
        }
        safe_indices.push(id);
        let src_base = row * feature_dim;
        safe_updates.extend_from_slice(&upstream[src_base..src_base + feature_dim]);
    }

    // Everything filtered → result is all zeros.
    if safe_indices.is_empty() {
        return Ok(vec![0.0_f32; out_len]);
    }

    let n_valid = safe_indices.len() as i32;
    let feature_i32 = feature_dim as i32;
    let vocab_i32 = vocab as i32;

    let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

    // Safety: `safe_updates` and `safe_indices` outlive the FFI call (the
    // C++ helper memcpy's into its own allocator-backed buffers). The
    // returned array is uniquely owned here and freed on every return path.
    unsafe {
        let out_arr = mlx_scatter_add_rows_f32(
            safe_updates.as_ptr(),
            safe_indices.as_ptr(),
            n_valid,
            feature_i32,
            vocab_i32,
        );
        if out_arr.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_scatter_add_rows_f32 returned null",
            ));
        }
        let host = match eval_and_readback(out_arr) {
            Ok(h) => h,
            Err(e) => {
                mlx_array_free(out_arr);
                return Err(e);
            }
        };
        mlx_array_free(out_arr);

        if host.len() != out_len {
            return Err(AutogradError::ShapeMismatch {
                expected: vec![out_len],
                got: vec![host.len()],
            });
        }
        Ok(host)
    }
}

// Shared tail: evaluate an MLX array, copy its contents into a freshly-
// allocated host vector, and return it. Caller holds `MLX_GUARD` and is
// responsible for freeing `arr` afterwards. Does not free on success or
// failure — the caller controls the array lifetime.
//
// Safety: `arr` must be a non-null pointer to a live MLX array owned by
// the caller for the duration of this call.
unsafe fn eval_and_readback(arr: *mut mlx_sys::mlx_array) -> Result<Vec<f32>> {
    unsafe {
        let mut eval_handles = [arr];
        mlx_eval(eval_handles.as_mut_ptr(), eval_handles.len());
        bump_eval_count();
        let size = mlx_array_size(arr);
        let data_ptr = mlx_array_data_float32(arr);
        if data_ptr.is_null() {
            return Err(AutogradError::TapeInvariant(
                "mlx_array_data_float32 returned null",
            ));
        }
        let mut host = vec![0.0f32; size];
        std::ptr::copy_nonoverlapping(data_ptr, host.as_mut_ptr(), size);
        Ok(host)
    }
}
