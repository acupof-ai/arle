//! Metal backend via mlx-sys. Host `Vec<f32>` stays authoritative: upload
//! into `mlx_array`, call `mlx_matmul`, `mlx_eval`, copy the result back.
//!
//! MLX's `mx::matmul` natively supports batched (rank-3) row-major inputs,
//! so we pass shape through unchanged. Shape validation mirrors
//! `cpu_matmul_forward` so the trait contract stays identical.

use crate::{AutogradError, Result, backend::Backend, backend::Device};
use mlx_sys::{
    MLX_FLOAT32, mlx_array, mlx_array_data_float32, mlx_array_free, mlx_array_from_data,
    mlx_array_size, mlx_eval, mlx_matmul,
};
use std::ffi::c_void;
use std::sync::Mutex;

// MLX's default stream/device is process-global and its C++ allocator is
// not re-entrant across threads. Concurrent `mlx_matmul` calls (e.g.
// default `cargo test` parallelism) SEGV the interpreter. A static
// mutex here is coarse but correct — training is single-threaded, and
// the lock is held only for the duration of one matmul FFI round-trip.
static MLX_GUARD: Mutex<()> = Mutex::new(());

#[derive(Debug, Default, Clone, Copy)]
pub struct MetalBackend;

impl Backend for MetalBackend {
    fn device(&self) -> Device {
        Device::Metal
    }

    fn matmul_forward(
        &self,
        a: &[f32],
        a_shape: &[usize],
        b: &[f32],
        b_shape: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let out_shape = match (a_shape.len(), b_shape.len()) {
            (2, 2) => {
                if a_shape[1] != b_shape[0] {
                    return Err(AutogradError::ShapeMismatch {
                        expected: vec![a_shape[1]],
                        got: vec![b_shape[0]],
                    });
                }
                vec![a_shape[0], b_shape[1]]
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
                vec![a_shape[0], a_shape[1], b_shape[2]]
            }
            _ => {
                return Err(AutogradError::InvalidRank {
                    expected: "both operands must be rank-2 or rank-3",
                    got: a_shape.len().max(b_shape.len()),
                });
            }
        };

        let a_shape_i32: Vec<i32> = a_shape.iter().map(|&d| d as i32).collect();
        let b_shape_i32: Vec<i32> = b_shape.iter().map(|&d| d as i32).collect();

        let _guard = MLX_GUARD.lock().expect("mlx guard poisoned");

        // Safety: all pointers are produced by mlx_array_from_data / mlx_matmul
        // and freed on every path. Host slices outlive the from_data call,
        // and MLX copies into its own storage before eval.
        let out = unsafe {
            let a_arr = mlx_array_from_data(
                a.as_ptr() as *const c_void,
                a_shape_i32.as_ptr(),
                a_shape_i32.len() as i32,
                MLX_FLOAT32,
            );
            if a_arr.is_null() {
                return Err(AutogradError::TapeInvariant(
                    "mlx_array_from_data returned null for A",
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
                    "mlx_array_from_data returned null for B",
                ));
            }

            let out_arr = mlx_matmul(a_arr, b_arr);
            if out_arr.is_null() {
                mlx_array_free(a_arr);
                mlx_array_free(b_arr);
                return Err(AutogradError::TapeInvariant("mlx_matmul returned null"));
            }

            let mut eval_arr: *mut mlx_array = out_arr;
            mlx_eval(&mut eval_arr, 1);

            let size = mlx_array_size(out_arr);
            let expected_size: usize = out_shape.iter().product();
            if size != expected_size {
                mlx_array_free(a_arr);
                mlx_array_free(b_arr);
                mlx_array_free(out_arr);
                return Err(AutogradError::ShapeMismatch {
                    expected: out_shape.clone(),
                    got: vec![size],
                });
            }

            let data_ptr = mlx_array_data_float32(out_arr);
            if data_ptr.is_null() {
                mlx_array_free(a_arr);
                mlx_array_free(b_arr);
                mlx_array_free(out_arr);
                return Err(AutogradError::TapeInvariant(
                    "mlx_array_data_float32 returned null",
                ));
            }

            let mut out_host = vec![0.0f32; size];
            std::ptr::copy_nonoverlapping(data_ptr, out_host.as_mut_ptr(), size);

            mlx_array_free(a_arr);
            mlx_array_free(b_arr);
            mlx_array_free(out_arr);

            out_host
        };

        Ok((out, out_shape))
    }
}
