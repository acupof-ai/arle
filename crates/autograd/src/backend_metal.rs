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
    MLX_FLOAT32, MLX_INT32, mlx_add, mlx_array_data_float32, mlx_array_free, mlx_array_from_data,
    mlx_array_new_float32, mlx_array_size, mlx_concatenate_axis, mlx_eval, mlx_exp,
    mlx_fast_rms_norm, mlx_logsumexp_axis, mlx_matmul, mlx_mean_axis, mlx_multiply, mlx_negative,
    mlx_scatter_add_rows_f32, mlx_sigmoid, mlx_slice, mlx_softmax_axis, mlx_subtract, mlx_sum_axis,
    mlx_take_axis, mlx_tanh,
};
use std::ffi::c_void;
use std::sync::Mutex;

// MLX's default stream/device is process-global and its C++ allocator is
// not re-entrant across threads. Concurrent `mlx_matmul` calls (e.g.
// default `cargo test` parallelism) SEGV the interpreter. A static
// mutex here is coarse but correct — training is single-threaded, and
// the lock is held only for the duration of one matmul FFI round-trip.
pub(crate) static MLX_GUARD: Mutex<()> = Mutex::new(());

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

    fn gather_last_dim_forward(
        &self,
        src: &[f32],
        src_shape: &[usize],
        ids: &[i32],
    ) -> Result<Vec<f32>> {
        mlx_gather_last_dim(src, src_shape, ids)
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
