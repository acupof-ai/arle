//! Thin MLX wrapper — direct mlx-sys (C API) bindings, no mlx-rs.
//!
//! Provides `MlxArray` (RAII newtype over `mlx_sys::mlx_array`) and thin
//! wrappers for the ~20 ops the Rust side needs (weight loading, KV cache
//! management, sampling, transforms).  All forward-pass computation lives
//! in C++ fused blocks.
//!
//! # Feature flag
//!
//! Gated behind `#[cfg(feature = "metal")]`.

#![allow(unsafe_code)]

use std::os::raw::{c_char, c_int, c_void};

use mlx_sys::{
    mlx_array, mlx_dtype, mlx_dtype__MLX_BFLOAT16, mlx_dtype__MLX_FLOAT16, mlx_dtype__MLX_FLOAT32,
    mlx_dtype__MLX_INT32, mlx_dtype__MLX_UINT8, mlx_dtype__MLX_UINT32, mlx_optional_float_,
    mlx_stream, mlx_vector_array,
};

// ── Dtype ────────────────────────────────────────────────────────────────────

/// Element type — mirrors `mlx_dtype` with a Rust enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Uint8,
    Uint32,
    Int32,
    Float16,
    Float32,
    Bfloat16,
}

impl Dtype {
    pub fn to_raw(self) -> mlx_dtype {
        match self {
            Dtype::Uint8 => mlx_dtype__MLX_UINT8,
            Dtype::Uint32 => mlx_dtype__MLX_UINT32,
            Dtype::Int32 => mlx_dtype__MLX_INT32,
            Dtype::Float16 => mlx_dtype__MLX_FLOAT16,
            Dtype::Float32 => mlx_dtype__MLX_FLOAT32,
            Dtype::Bfloat16 => mlx_dtype__MLX_BFLOAT16,
        }
    }

    pub fn from_raw(raw: mlx_dtype) -> Option<Self> {
        match raw {
            mlx_dtype__MLX_UINT8 => Some(Dtype::Uint8),
            mlx_dtype__MLX_UINT32 => Some(Dtype::Uint32),
            mlx_dtype__MLX_INT32 => Some(Dtype::Int32),
            mlx_dtype__MLX_FLOAT16 => Some(Dtype::Float16),
            mlx_dtype__MLX_FLOAT32 => Some(Dtype::Float32),
            mlx_dtype__MLX_BFLOAT16 => Some(Dtype::Bfloat16),
            _ => None,
        }
    }
}

// ── Default stream ───────────────────────────────────────────────────────────

/// Sync-safe wrapper for `mlx_stream` (contains a raw pointer that Rust
/// won't auto-impl Send/Sync for, but MLX streams are thread-safe).
struct SyncStreamAndDevice {
    stream: mlx_stream,
    _device: mlx_sys::mlx_device, // keep device alive alongside stream
}
unsafe impl Send for SyncStreamAndDevice {}
unsafe impl Sync for SyncStreamAndDevice {}

/// Return the default GPU stream. Cached on first call.
/// Expose the cached default stream for internal use (e.g., loader).
pub(crate) fn default_stream_raw() -> mlx_stream {
    // Use a separate generation stream for better GPU pipelining,
    // matching mlx_lm's `generation_stream = mx.new_stream(mx.default_device())`.
    generation_stream()
}

/// Separate stream for generation — enables GPU pipelining within forward pass.
fn generation_stream() -> mlx_stream {
    static CACHED: std::sync::LazyLock<SyncStreamAndDevice> = std::sync::LazyLock::new(|| unsafe {
        let dev = mlx_sys::mlx_device_new_type(mlx_sys::mlx_device_type__MLX_GPU, 0);
        let stream = mlx_sys::mlx_stream_new_device(dev);
        SyncStreamAndDevice {
            stream,
            _device: dev,
        }
    });
    CACHED.stream
}

fn default_stream() -> mlx_stream {
    static CACHED: std::sync::LazyLock<SyncStreamAndDevice> = std::sync::LazyLock::new(|| unsafe {
        let dev = mlx_sys::mlx_device_new_type(mlx_sys::mlx_device_type__MLX_GPU, 0);
        let mut stream = mlx_sys::mlx_stream_new();
        mlx_sys::mlx_get_default_stream(&mut stream, dev);
        SyncStreamAndDevice {
            stream,
            _device: dev,
        }
    });
    CACHED.stream
}

// ── MlxArray ─────────────────────────────────────────────────────────────────

/// RAII wrapper over `mlx_sys::mlx_array`.
///
/// `#[repr(transparent)]` so `&MlxArray` can be transmuted to `&mlx_array`
/// for zero-cost FFI.  Drop calls `mlx_array_free` to decrement the
/// internal reference count.
#[repr(transparent)]
pub struct MlxArray(mlx_array);

impl std::fmt::Debug for MlxArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MlxArray({:?}, {:?})", self.shape(), self.dtype())
    }
}

impl Drop for MlxArray {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_array_free(self.0);
        }
    }
}

impl Clone for MlxArray {
    fn clone(&self) -> Self {
        let mut new = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_array_set(&mut new, self.0);
        }
        Self(new)
    }
}

// SAFETY: MLX manages its own internal locking; Arrays are safe to send
// across threads (same guarantee as mlx-rs).
unsafe impl Send for MlxArray {}
unsafe impl Sync for MlxArray {}

impl MlxArray {
    // ── Ownership helpers ────────────────────────────────────────────────

    /// Wrap a raw `mlx_array` handle, taking ownership.
    /// The caller must not free the handle afterwards.
    pub unsafe fn from_raw(raw: mlx_array) -> Self {
        Self(raw)
    }

    /// Borrow the raw handle (no ownership transfer).
    pub fn as_raw(&self) -> mlx_array {
        self.0
    }

    /// Consume self, returning the raw handle without calling `mlx_array_free`.
    pub fn into_raw(self) -> mlx_array {
        let raw = self.0;
        std::mem::forget(self);
        raw
    }

    /// Access the inner handle as a mutable pointer — for FFI out-params
    /// that update the array in-place (e.g. KV cache slice_update).
    pub fn as_raw_mut(&mut self) -> *mut mlx_array {
        &mut self.0 as *mut mlx_array
    }

    // ── Construction ─────────────────────────────────────────────────────

    /// Create an array from a raw data pointer (copies the data).
    pub unsafe fn from_raw_data(data: *const c_void, shape: &[i32], dtype: Dtype) -> Self {
        let raw = unsafe {
            mlx_sys::mlx_array_new_data(data, shape.as_ptr(), shape.len() as c_int, dtype.to_raw())
        };
        Self(raw)
    }

    /// Create an array from a Rust slice.
    pub fn from_slice_i32(data: &[i32], shape: &[i32]) -> Self {
        unsafe { Self::from_raw_data(data.as_ptr().cast(), shape, Dtype::Int32) }
    }

    /// Create an array from a Rust f32 slice.
    pub fn from_slice_f32(data: &[f32], shape: &[i32]) -> Self {
        unsafe { Self::from_raw_data(data.as_ptr().cast(), shape, Dtype::Float32) }
    }

    /// Create a scalar f32 array.
    pub fn scalar_f32(val: f32) -> Self {
        Self(unsafe { mlx_sys::mlx_array_new_float32(val) })
    }

    /// Create a scalar i32 array (shape []).
    pub fn scalar_i32(val: i32) -> Self {
        Self(unsafe { mlx_sys::mlx_array_new_int(val) })
    }

    // ── Inspection ───────────────────────────────────────────────────────

    pub fn ndim(&self) -> usize {
        unsafe { mlx_sys::mlx_array_ndim(self.0) }
    }

    pub fn shape(&self) -> &[i32] {
        unsafe {
            let ptr = mlx_sys::mlx_array_shape(self.0);
            let ndim = self.ndim();
            if ndim == 0 || ptr.is_null() {
                &[]
            } else {
                std::slice::from_raw_parts(ptr, ndim)
            }
        }
    }

    pub fn dtype(&self) -> Dtype {
        let raw = unsafe { mlx_sys::mlx_array_dtype(self.0) };
        Dtype::from_raw(raw).expect("unknown mlx_dtype")
    }

    /// Extract a scalar i32 value (blocks until computed).
    pub fn item_i32(&self) -> i32 {
        let mut val = 0i32;
        unsafe {
            mlx_sys::mlx_array_item_int32(&mut val, self.0);
        }
        val
    }

    /// Extract a scalar f32 value (blocks until computed).
    pub fn item_f32(&self) -> f32 {
        let mut val = 0.0f32;
        unsafe {
            mlx_sys::mlx_array_item_float32(&mut val, self.0);
        }
        val
    }

    /// Access the underlying data pointer (after eval).
    pub fn as_slice_f32(&self) -> &[f32] {
        unsafe {
            let ptr = mlx_sys::mlx_array_data_float32(self.0);
            let size = mlx_sys::mlx_array_size(self.0);
            std::slice::from_raw_parts(ptr, size)
        }
    }
}

// ── Ops ──────────────────────────────────────────────────────────────────────

/// Helper: call a binary mlx op, return result.
macro_rules! mlx_binary_op {
    ($name:ident, $cfn:ident) => {
        pub fn $name(a: &MlxArray, b: &MlxArray) -> MlxArray {
            let mut res = unsafe { mlx_sys::mlx_array_new() };
            unsafe {
                mlx_sys::$cfn(&mut res, a.0, b.0, default_stream());
            }
            MlxArray(res)
        }
    };
}

mlx_binary_op!(add, mlx_add);
mlx_binary_op!(subtract, mlx_subtract);
mlx_binary_op!(multiply, mlx_multiply);
mlx_binary_op!(matmul, mlx_matmul);
mlx_binary_op!(greater, mlx_greater);

/// Helper: call a unary mlx op, return result.
macro_rules! mlx_unary_op {
    ($name:ident, $cfn:ident) => {
        pub fn $name(a: &MlxArray) -> MlxArray {
            let mut res = unsafe { mlx_sys::mlx_array_new() };
            unsafe {
                mlx_sys::$cfn(&mut res, a.0, default_stream());
            }
            MlxArray(res)
        }
    };
}

mlx_unary_op!(exp, mlx_exp);
mlx_unary_op!(log1p, mlx_log1p);
mlx_unary_op!(negative, mlx_negative);
mlx_unary_op!(sqrt, mlx_sqrt);
mlx_unary_op!(reciprocal, mlx_reciprocal);

pub fn transpose_axes(a: &MlxArray, axes: &[i32]) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_transpose_axes(&mut res, a.0, axes.as_ptr(), axes.len(), default_stream());
    }
    MlxArray(res)
}

pub fn reshape(a: &MlxArray, shape: &[i32]) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_reshape(&mut res, a.0, shape.as_ptr(), shape.len(), default_stream());
    }
    MlxArray(res)
}

pub fn transpose_all(a: &MlxArray) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_transpose(&mut res, a.0, default_stream());
    }
    MlxArray(res)
}

pub fn as_dtype(a: &MlxArray, dtype: Dtype) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_astype(&mut res, a.0, dtype.to_raw(), default_stream());
    }
    MlxArray(res)
}

pub fn zeros(shape: &[i32], dtype: Dtype) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_zeros(
            &mut res,
            shape.as_ptr(),
            shape.len(),
            dtype.to_raw(),
            default_stream(),
        );
    }
    MlxArray(res)
}

pub fn take_axis(a: &MlxArray, indices: &MlxArray, axis: i32) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_take_axis(&mut res, a.0, indices.0, axis as c_int, default_stream());
    }
    MlxArray(res)
}

pub fn concatenate_axis(arrays: &[MlxArray], axis: i32) -> MlxArray {
    let raw_arrays: Vec<mlx_array> = arrays.iter().map(|a| a.0).collect();
    let vec = unsafe { mlx_sys::mlx_vector_array_new_data(raw_arrays.as_ptr(), raw_arrays.len()) };
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_concatenate_axis(&mut res, vec, axis as c_int, default_stream());
        mlx_sys::mlx_vector_array_free(vec);
    }
    MlxArray(res)
}

pub fn broadcast_to(a: &MlxArray, shape: &[i32]) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_broadcast_to(&mut res, a.0, shape.as_ptr(), shape.len(), default_stream());
    }
    MlxArray(res)
}

pub fn sigmoid(a: &MlxArray) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_sigmoid(&mut res, a.0, default_stream());
    }
    MlxArray(res)
}

/// SiLU activation: silu(x) = x * sigmoid(x).
pub fn silu(a: &MlxArray) -> MlxArray {
    multiply(a, &sigmoid(a))
}

pub fn sum_axis(a: &MlxArray, axis: i32, keepdims: bool) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_sum_axis(&mut res, a.0, axis as c_int, keepdims, default_stream());
    }
    MlxArray(res)
}

pub fn expand_dims(a: &MlxArray, axis: i32) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_expand_dims(&mut res, a.0, axis as c_int, default_stream());
    }
    MlxArray(res)
}

/// Element-wise conditional: where(mask, a, b) — selects from `a` where mask is true, else `b`.
pub fn where_(condition: &MlxArray, a: &MlxArray, b: &MlxArray) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_where(&mut res, condition.0, a.0, b.0, default_stream());
    }
    MlxArray(res)
}

/// Multi-axis slice: `a[start[0]:stop[0]:strides[0], start[1]:stop[1]:strides[1], ...]`.
/// Pass empty strides for default stride of 1.
pub fn slice(a: &MlxArray, start: &[i32], stop: &[i32], strides: &[i32]) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_slice(
            &mut res,
            a.0,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            default_stream(),
        );
    }
    MlxArray(res)
}

pub fn argmax(a: &MlxArray) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_argmax(&mut res, a.0, false, default_stream());
    }
    MlxArray(res)
}

pub fn slice_update(a: &mut MlxArray, update: &MlxArray, start: &[i32], stop: &[i32]) -> MlxArray {
    // Default strides: all 1s, matching the number of start/stop indices.
    let strides: Vec<i32> = vec![1i32; start.len()];
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_slice_update(
            &mut res,
            a.0,
            update.0,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            default_stream(),
        );
    }
    MlxArray(res)
}

pub fn dequantize(
    w: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    group_size: i32,
    bits: i32,
) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_dequantize(
            &mut res,
            w.0,
            scales.0,
            biases.0,
            group_size as c_int,
            bits as c_int,
            default_stream(),
        );
    }
    MlxArray(res)
}

pub fn quantized_matmul(
    x: &MlxArray,
    w: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    transpose: bool,
    group_size: i32,
    bits: i32,
) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_quantized_matmul(
            &mut res,
            x.0,
            w.0,
            scales.0,
            biases.0,
            transpose,
            group_size as c_int,
            bits as c_int,
            default_stream(),
        );
    }
    MlxArray(res)
}

/// Depthwise conv1d: input[batch, seq, channels], weight[out, kernel, 1] for groups=out.
pub fn conv1d(
    input: &MlxArray,
    weight: &MlxArray,
    stride: i32,
    padding: i32,
    groups: i32,
) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_conv1d(
            &mut res,
            input.0,
            weight.0,
            stride,
            padding,
            1, /*dilation*/
            groups,
            default_stream(),
        );
    }
    MlxArray(res)
}

// ── Fast ops ─────────────────────────────────────────────────────────────────

pub fn rms_norm(x: &MlxArray, weight: &MlxArray, eps: f32) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_fast_rms_norm(&mut res, x.0, weight.0, eps, default_stream());
    }
    MlxArray(res)
}

pub fn rope(
    x: &MlxArray,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: i32,
) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    let opt_base = mlx_optional_float_ {
        value: base,
        has_value: true,
    };
    // No custom freqs — pass a null/empty array.
    let no_freqs = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_fast_rope(
            &mut res,
            x.0,
            dims as c_int,
            traditional,
            opt_base,
            scale,
            offset as c_int,
            no_freqs,
            default_stream(),
        );
        mlx_sys::mlx_array_free(no_freqs);
    }
    MlxArray(res)
}

pub fn scaled_dot_product_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: Option<&str>,
) -> MlxArray {
    use std::ffi::CString;
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    // mask_mode must never be NULL — std::string(nullptr) is UB in the C wrapper.
    let mask_cstr = CString::new(mask.unwrap_or("")).unwrap();
    let mask_ptr = mask_cstr.as_ptr();
    // Empty mask vector — mask_mode string handles causal masking.
    let empty_mask_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    unsafe {
        mlx_sys::mlx_fast_scaled_dot_product_attention(
            &mut res,
            q.0,
            k.0,
            v.0,
            scale,
            mask_ptr,
            empty_mask_vec,
            default_stream(),
        );
        mlx_sys::mlx_vector_array_free(empty_mask_vec);
    }
    MlxArray(res)
}

// ── Sampling ─────────────────────────────────────────────────────────────────

pub fn categorical(logits: &MlxArray) -> MlxArray {
    let mut res = unsafe { mlx_sys::mlx_array_new() };
    let no_key = unsafe { mlx_sys::mlx_array_new() };
    unsafe {
        mlx_sys::mlx_random_categorical(
            &mut res,
            logits.0,
            -1, // axis (last)
            no_key,
            default_stream(),
        );
        mlx_sys::mlx_array_free(no_key);
    }
    MlxArray(res)
}

// ── Transforms ───────────────────────────────────────────────────────────────

/// Synchronously evaluate arrays (materialize lazy compute graph).
pub fn eval(arrays: &[&MlxArray]) -> c_int {
    let raw: Vec<mlx_array> = arrays.iter().map(|a| a.0).collect();
    let vec = unsafe { mlx_sys::mlx_vector_array_new_data(raw.as_ptr(), raw.len()) };
    let ret = unsafe { mlx_sys::mlx_eval(vec) };
    unsafe {
        mlx_sys::mlx_vector_array_free(vec);
    }
    ret
}

/// Asynchronously schedule evaluation (non-blocking).
pub fn async_eval(arrays: &[&MlxArray]) -> c_int {
    let raw: Vec<mlx_array> = arrays.iter().map(|a| a.0).collect();
    let vec = unsafe { mlx_sys::mlx_vector_array_new_data(raw.as_ptr(), raw.len()) };
    let ret = unsafe { mlx_sys::mlx_async_eval(vec) };
    unsafe {
        mlx_sys::mlx_vector_array_free(vec);
    }
    ret
}

/// Clear the MLX compilation / kernel cache.
pub fn compile_clear_cache() {
    unsafe {
        mlx_sys::mlx_detail_compile_clear_cache();
    }
}

// ── Metal Kernel ─────────────────────────────────────────────────────────────

/// A custom Metal kernel created via `mx.fast.metal_kernel()`.
pub struct MetalKernel(mlx_sys::mlx_fast_metal_kernel);

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_free(self.0);
        }
    }
}
unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl MetalKernel {
    /// Create a new Metal kernel from source.
    pub fn new(name: &str, input_names: &[&str], output_names: &[&str], source: &str) -> Self {
        use std::ffi::CString;
        let name_c = CString::new(name).unwrap();
        let source_c = CString::new(source).unwrap();
        let input_cstrs: Vec<CString> = input_names
            .iter()
            .map(|s| CString::new(*s).unwrap())
            .collect();
        let input_ptrs: Vec<*const c_char> = input_cstrs.iter().map(|c| c.as_ptr()).collect();
        let output_cstrs: Vec<CString> = output_names
            .iter()
            .map(|s| CString::new(*s).unwrap())
            .collect();
        let output_ptrs: Vec<*const c_char> = output_cstrs.iter().map(|c| c.as_ptr()).collect();

        unsafe {
            let in_vec = mlx_sys::mlx_vector_string_new_data(
                input_ptrs.as_ptr() as *mut _,
                input_ptrs.len(),
            );
            let out_vec = mlx_sys::mlx_vector_string_new_data(
                output_ptrs.as_ptr() as *mut _,
                output_ptrs.len(),
            );
            let kernel = mlx_sys::mlx_fast_metal_kernel_new(
                name_c.as_ptr(),
                in_vec,
                out_vec,
                source_c.as_ptr(),
                std::ptr::null(), // header
                true,             // ensure_row_contiguous
                false,            // atomic_outputs
            );
            mlx_sys::mlx_vector_string_free(in_vec);
            mlx_sys::mlx_vector_string_free(out_vec);
            Self(kernel)
        }
    }

    /// Execute the kernel with given inputs and config.
    pub fn apply(
        &self,
        inputs: &[&MlxArray],
        grid: [i32; 3],
        threadgroup: [i32; 3],
        output_shapes: &[&[i32]],
        output_dtypes: &[Dtype],
        template_int_args: &[(&str, i32)],
        template_dtype_args: &[(&str, Dtype)],
    ) -> Vec<MlxArray> {
        use std::ffi::CString;

        let raw_in: Vec<mlx_array> = inputs.iter().map(|a| a.0).collect();
        let in_vec = unsafe { mlx_sys::mlx_vector_array_new_data(raw_in.as_ptr(), raw_in.len()) };

        let config = unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() };
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, grid[0], grid[1], grid[2]);
            mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(
                config,
                threadgroup[0],
                threadgroup[1],
                threadgroup[2],
            );

            for (shape, dtype) in output_shapes.iter().zip(output_dtypes.iter()) {
                mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                    config,
                    shape.as_ptr(),
                    shape.len(),
                    dtype.to_raw(),
                );
            }

            for (name, val) in template_int_args {
                let name_c = CString::new(*name).unwrap();
                mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                    config,
                    name_c.as_ptr(),
                    *val,
                );
            }
            for (name, dtype) in template_dtype_args {
                let name_c = CString::new(*name).unwrap();
                mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                    config,
                    name_c.as_ptr(),
                    dtype.to_raw(),
                );
            }
        }

        let mut out_vec = unsafe { mlx_sys::mlx_vector_array_new() };
        let ret = unsafe {
            mlx_sys::mlx_fast_metal_kernel_apply(
                &mut out_vec,
                self.0,
                in_vec,
                config,
                default_stream(),
            )
        };
        if ret != 0 {
            eprintln!("[MLX] metal_kernel_apply failed with code {ret}");
        }
        unsafe {
            mlx_sys::mlx_vector_array_free(in_vec);
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
        }

        let n = unsafe { mlx_sys::mlx_vector_array_size(out_vec) };
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let mut arr = unsafe { mlx_sys::mlx_array_new() };
            unsafe {
                mlx_sys::mlx_vector_array_get(&mut arr, out_vec, i);
            }
            results.push(MlxArray(arr));
        }
        unsafe {
            mlx_sys::mlx_vector_array_free(out_vec);
        }
        results
    }
}

// ── Compile ──────────────────────────────────────────────────────────────────

/// A compiled MLX closure — wraps `mlx_closure` with RAII.
pub struct CompiledFn(mlx_sys::mlx_closure);

impl Drop for CompiledFn {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_closure_free(self.0);
        }
    }
}

// SAFETY: MLX closures are thread-safe (same as Arrays).
unsafe impl Send for CompiledFn {}
unsafe impl Sync for CompiledFn {}

impl CompiledFn {
    /// Call the compiled function: `inputs → outputs`.
    pub fn call(&self, inputs: &[&MlxArray]) -> Vec<MlxArray> {
        let raw_in: Vec<mlx_array> = inputs.iter().map(|a| a.0).collect();
        let in_vec = unsafe { mlx_sys::mlx_vector_array_new_data(raw_in.as_ptr(), raw_in.len()) };
        let mut out_vec = unsafe { mlx_sys::mlx_vector_array_new() };
        unsafe {
            mlx_sys::mlx_closure_apply(&mut out_vec, self.0, in_vec);
            mlx_sys::mlx_vector_array_free(in_vec);
        }
        // Extract output arrays.
        let n = unsafe { mlx_sys::mlx_vector_array_size(out_vec) };
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let mut arr = unsafe { mlx_sys::mlx_array_new() };
            unsafe {
                mlx_sys::mlx_vector_array_get(&mut arr, out_vec, i);
            }
            results.push(MlxArray(arr));
        }
        unsafe {
            mlx_sys::mlx_vector_array_free(out_vec);
        }
        results
    }
}

/// Compile a function `fn(vector_array) -> vector_array` with optional shapeless mode.
///
/// The callback `f` receives `(result_out, inputs, payload)`. It must populate
/// `result_out` with the output vector_array and return 0 on success.
///
/// `payload` is an opaque pointer passed to every call (e.g. weights).
/// `dtor` is called when the closure is freed.
pub fn compile_fn(
    f: unsafe extern "C" fn(
        *mut mlx_sys::mlx_vector_array,
        mlx_sys::mlx_vector_array,
        *mut std::os::raw::c_void,
    ) -> c_int,
    payload: *mut std::os::raw::c_void,
    dtor: Option<unsafe extern "C" fn(*mut std::os::raw::c_void)>,
    shapeless: bool,
) -> CompiledFn {
    unsafe {
        let closure = mlx_sys::mlx_closure_new_func_payload(Some(f), payload, dtor);
        let mut compiled = mlx_sys::mlx_closure_new();
        mlx_sys::mlx_compile(&mut compiled, closure, shapeless);
        mlx_sys::mlx_closure_free(closure); // original closure no longer needed
        CompiledFn(compiled)
    }
}

/// Compile a simple function (no payload) with shapeless mode.
pub fn compile_simple(
    f: unsafe extern "C" fn(*mut mlx_sys::mlx_vector_array, mlx_sys::mlx_vector_array) -> c_int,
    shapeless: bool,
) -> CompiledFn {
    unsafe {
        let closure = mlx_sys::mlx_closure_new_func(Some(f));
        let mut compiled = mlx_sys::mlx_closure_new();
        mlx_sys::mlx_compile(&mut compiled, closure, shapeless);
        mlx_sys::mlx_closure_free(closure);
        CompiledFn(compiled)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_array_lifecycle() {
        let a = MlxArray::from_slice_f32(&[1.0, 2.0, 3.0], &[3]);
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.dtype(), Dtype::Float32);
    }

    #[test]
    fn mlx_array_clone_shares_data() {
        let a = MlxArray::from_slice_f32(&[1.0, 2.0], &[2]);
        let b = a.clone();
        assert_eq!(b.shape(), a.shape());
        assert_eq!(b.dtype(), a.dtype());
    }

    #[test]
    fn mlx_array_scalar_item() {
        let a = MlxArray::scalar_f32(42.0);
        eval(&[&a]);
        assert!((a.item_f32() - 42.0).abs() < 1e-6);
    }

    #[test]
    fn mlx_add_basic() {
        let a = MlxArray::from_slice_f32(&[1.0, 2.0], &[2]);
        let b = MlxArray::from_slice_f32(&[3.0, 4.0], &[2]);
        let c = add(&a, &b);
        eval(&[&c]);
        let vals = c.as_slice_f32();
        assert!((vals[0] - 4.0).abs() < 1e-6);
        assert!((vals[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn mlx_zeros_and_reshape() {
        let a = zeros(&[2, 3], Dtype::Float32);
        assert_eq!(a.shape(), &[2, 3]);
        let b = reshape(&a, &[6]);
        assert_eq!(b.shape(), &[6]);
    }

    #[test]
    fn mlx_matmul_basic() {
        // [1,2] @ [2,1] = [[5]]
        let a = MlxArray::from_slice_f32(&[1.0, 2.0], &[1, 2]);
        let b = MlxArray::from_slice_f32(&[1.0, 2.0], &[2, 1]);
        let c = matmul(&a, &b);
        eval(&[&c]);
        assert_eq!(c.shape(), &[1, 1]);
        assert!((c.as_slice_f32()[0] - 5.0).abs() < 1e-6);
    }
}
