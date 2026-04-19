//! Metal backend via mlx-sys. Host `Vec<f32>` stays authoritative: upload
//! into `mlx_array`, call `mlx_matmul`, `mlx_eval`, copy the result back.
//!
//! MLX's `mx::matmul` natively supports batched (rank-3) row-major inputs,
//! so we pass shape through unchanged. Shape validation mirrors
//! `cpu_matmul_forward` so the trait contract stays identical.

use crate::{
    AutogradError, Result,
    backend::{Backend, Device, DeviceHandle, MlxHandle, matmul_output_shape},
};
use mlx_sys::{
    MLX_FLOAT32, mlx_array_data_float32, mlx_array_from_data, mlx_array_size, mlx_eval, mlx_matmul,
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
}
