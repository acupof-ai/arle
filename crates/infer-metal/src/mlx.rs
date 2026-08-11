//! Minimal MLX wrapper for the Metal executor.
//!
//! Enough to load MLX safetensors, register weights with the Qwen3.5 C++
//! compiled model, allocate session KV/GDR arrays, and sample. Device tensors
//! stay below the executor seam.

use std::ffi::CStr;
use std::os::raw::c_void;

pub fn check_mlx_error() -> anyhow::Result<()> {
    // SAFETY: mlx_last_error returned a non-null NUL-terminated bridge-owned string, read here before any further FFI.
    unsafe {
        let ptr = mlx_sys::mlx_last_error();
        if ptr.is_null() {
            Ok(())
        } else {
            let msg = CStr::from_ptr(ptr).to_string_lossy();
            Err(anyhow::anyhow!("MLX error: {msg}"))
        }
    }
}

fn mlx_error_message() -> Option<String> {
    // SAFETY: mlx_last_error returned a non-null NUL-terminated bridge-owned string, read here before any further FFI.
    unsafe {
        let ptr = mlx_sys::mlx_last_error();
        (!ptr.is_null()).then(|| CStr::from_ptr(ptr).to_string_lossy().into_owned())
    }
}

fn panic_if_mlx_error(op: &str) {
    if let Some(msg) = mlx_error_message() {
        panic!("{op} failed: {msg}");
    }
}

fn mlx_array_from_raw_or_panic(raw: *mut mlx_sys::mlx_array, op: &str) -> MlxArray {
    if raw.is_null() {
        match mlx_error_message() {
            Some(msg) => panic!("{op} returned a null MLX handle: {msg}"),
            None => panic!("{op} returned a null MLX handle"),
        }
    }
    MlxArray(raw)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    Bfloat16,
    Complex64,
}

impl Dtype {
    pub fn to_raw(self) -> i32 {
        match self {
            Dtype::Bool => mlx_sys::MLX_BOOL,
            Dtype::Uint8 => mlx_sys::MLX_UINT8,
            Dtype::Uint16 => mlx_sys::MLX_UINT16,
            Dtype::Uint32 => mlx_sys::MLX_UINT32,
            Dtype::Uint64 => mlx_sys::MLX_UINT64,
            Dtype::Int8 => mlx_sys::MLX_INT8,
            Dtype::Int16 => mlx_sys::MLX_INT16,
            Dtype::Int32 => mlx_sys::MLX_INT32,
            Dtype::Int64 => mlx_sys::MLX_INT64,
            Dtype::Float16 => mlx_sys::MLX_FLOAT16,
            Dtype::Float32 => mlx_sys::MLX_FLOAT32,
            Dtype::Float64 => mlx_sys::MLX_FLOAT64,
            Dtype::Bfloat16 => mlx_sys::MLX_BFLOAT16,
            Dtype::Complex64 => mlx_sys::MLX_COMPLEX64,
        }
    }

    pub fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            x if x == mlx_sys::MLX_BOOL => Some(Dtype::Bool),
            x if x == mlx_sys::MLX_UINT8 => Some(Dtype::Uint8),
            x if x == mlx_sys::MLX_UINT16 => Some(Dtype::Uint16),
            x if x == mlx_sys::MLX_UINT32 => Some(Dtype::Uint32),
            x if x == mlx_sys::MLX_UINT64 => Some(Dtype::Uint64),
            x if x == mlx_sys::MLX_INT8 => Some(Dtype::Int8),
            x if x == mlx_sys::MLX_INT16 => Some(Dtype::Int16),
            x if x == mlx_sys::MLX_INT32 => Some(Dtype::Int32),
            x if x == mlx_sys::MLX_INT64 => Some(Dtype::Int64),
            x if x == mlx_sys::MLX_FLOAT16 => Some(Dtype::Float16),
            x if x == mlx_sys::MLX_FLOAT32 => Some(Dtype::Float32),
            x if x == mlx_sys::MLX_FLOAT64 => Some(Dtype::Float64),
            x if x == mlx_sys::MLX_BFLOAT16 => Some(Dtype::Bfloat16),
            x if x == mlx_sys::MLX_COMPLEX64 => Some(Dtype::Complex64),
            _ => None,
        }
    }
}

pub struct MlxArray(*mut mlx_sys::mlx_array);

impl Drop for MlxArray {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: self.0 is the owned handle, freed exactly once here.
            unsafe {
                mlx_sys::mlx_array_free(self.0);
            }
        }
    }
}

impl Clone for MlxArray {
    fn clone(&self) -> Self {
        mlx_array_from_raw_or_panic(
            // SAFETY: the source is a valid owned handle; mlx_array_clone bumps its refcount.
            unsafe { mlx_sys::mlx_array_clone(self.0) },
            "mlx_array_clone",
        )
    }
}

impl std::fmt::Debug for MlxArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MlxArray({:?}, {:?})", self.shape(), self.dtype())
    }
}

// SAFETY: MlxArray solely owns its handle and MLX process-global state is serialized via mlx_guard, so the handle may cross threads.
unsafe impl Send for MlxArray {}

impl MlxArray {
    /// # Safety
    /// `raw` must be a valid owned MLX array handle returned by the bridge.
    pub unsafe fn from_raw(raw: *mut mlx_sys::mlx_array) -> Self {
        mlx_array_from_raw_or_panic(raw, "MlxArray::from_raw")
    }

    pub fn as_raw(&self) -> *mut mlx_sys::mlx_array {
        self.0
    }

    /// # Safety
    /// `data` must remain valid for MLX to read for the duration required by
    /// the bridge call.
    pub unsafe fn from_raw_data(data: *const c_void, shape: &[i32], dtype: Dtype) -> Self {
        mlx_array_from_raw_or_panic(
            // SAFETY: `data` stays valid for the bridge read (guaranteed by the enclosing unsafe fn's contract).
            unsafe {
                mlx_sys::mlx_array_from_data(
                    data,
                    shape.as_ptr(),
                    shape.len() as i32,
                    dtype.to_raw(),
                )
            },
            "mlx_array_from_data",
        )
    }

    pub fn from_slice_i32(data: &[i32], shape: &[i32]) -> Self {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe { Self::from_raw_data(data.as_ptr().cast(), shape, Dtype::Int32) }
    }

    pub fn from_bytes(data: &[u8], shape: &[i32], dtype: Dtype) -> Self {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe { Self::from_raw_data(data.as_ptr().cast(), shape, dtype) }
    }

    pub fn scalar_f32(value: f32) -> Self {
        mlx_array_from_raw_or_panic(
            // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
            unsafe { mlx_sys::mlx_array_new_float32(value) },
            "mlx_array_new_float32",
        )
    }

    pub fn ndim(&self) -> usize {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let ndim = unsafe { mlx_sys::mlx_array_ndim(self.0) as usize };
        panic_if_mlx_error("mlx_array_ndim");
        ndim
    }

    pub fn shape(&self) -> &[i32] {
        // SAFETY: the pointer addresses MLX-owned contiguous data of the returned length, valid while &self is borrowed.
        unsafe {
            let ptr = mlx_sys::mlx_array_shape(self.0);
            let n = self.ndim();
            if ptr.is_null() && n > 0 {
                panic_if_mlx_error("mlx_array_shape");
                panic!("mlx_array_shape returned null for non-scalar array");
            }
            panic_if_mlx_error("mlx_array_shape");
            if n == 0 || ptr.is_null() {
                &[]
            } else {
                std::slice::from_raw_parts(ptr, n)
            }
        }
    }

    pub fn dtype(&self) -> Dtype {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let raw = unsafe { mlx_sys::mlx_array_dtype(self.0) };
        panic_if_mlx_error("mlx_array_dtype");
        Dtype::from_raw(raw)
            .unwrap_or_else(|| panic!("mlx_array_dtype returned unknown dtype {raw}"))
    }

    pub fn nbytes(&self) -> usize {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let bytes = unsafe { mlx_sys::mlx_array_nbytes(self.0) };
        panic_if_mlx_error("mlx_array_nbytes");
        bytes
    }

    pub fn export_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.nbytes()];
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let written = unsafe {
            mlx_sys::mlx_array_export_bytes(self.0, bytes.as_mut_ptr().cast(), bytes.len())
        };
        panic_if_mlx_error("mlx_array_export_bytes");
        assert_eq!(
            written,
            bytes.len(),
            "mlx_array_export_bytes wrote a different byte count"
        );
        bytes
    }

    pub fn item_i32(&self) -> i32 {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        let value = unsafe { mlx_sys::mlx_array_item_int32(self.0) };
        panic_if_mlx_error("mlx_array_item_int32");
        value
    }

    pub fn as_slice_f32(&self) -> &[f32] {
        // SAFETY: the pointer addresses MLX-owned contiguous data of the returned length, valid while &self is borrowed.
        unsafe {
            let ptr = mlx_sys::mlx_array_data_float32(self.0);
            let len = mlx_sys::mlx_array_size(self.0);
            if ptr.is_null() && len > 0 {
                panic_if_mlx_error("mlx_array_data_float32");
                panic!("mlx_array_data_float32 returned null for non-empty array");
            }
            panic_if_mlx_error("mlx_array_data_float32");
            std::slice::from_raw_parts(ptr, len)
        }
    }

    pub fn as_slice_i32(&self) -> &[i32] {
        // SAFETY: the pointer addresses MLX-owned contiguous data of the returned length, valid while &self is borrowed.
        unsafe {
            let ptr = mlx_sys::mlx_array_data_int32(self.0);
            let len = mlx_sys::mlx_array_size(self.0);
            if ptr.is_null() && len > 0 {
                panic_if_mlx_error("mlx_array_data_int32");
                panic!("mlx_array_data_int32 returned null for non-empty array");
            }
            panic_if_mlx_error("mlx_array_data_int32");
            std::slice::from_raw_parts(ptr, len)
        }
    }
}

macro_rules! binary_op {
    ($name:ident, $cfn:ident) => {
        pub fn $name(a: &MlxArray, b: &MlxArray) -> MlxArray {
            // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
            mlx_array_from_raw_or_panic(unsafe { mlx_sys::$cfn(a.0, b.0) }, stringify!($cfn))
        }
    };
}

binary_op!(add, mlx_add);
binary_op!(matmul, mlx_matmul);

pub fn reshape(a: &MlxArray, shape: &[i32]) -> MlxArray {
    mlx_array_from_raw_or_panic(
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe { mlx_sys::mlx_reshape(a.0, shape.as_ptr(), shape.len()) },
        "mlx_reshape",
    )
}

pub fn transpose_all(a: &MlxArray) -> MlxArray {
    // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
    mlx_array_from_raw_or_panic(unsafe { mlx_sys::mlx_transpose(a.0) }, "mlx_transpose")
}

pub fn transpose_axes(a: &MlxArray, axes: &[i32]) -> MlxArray {
    mlx_array_from_raw_or_panic(
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe { mlx_sys::mlx_transpose_axes(a.0, axes.as_ptr(), axes.len()) },
        "mlx_transpose_axes",
    )
}

pub fn as_dtype(a: &MlxArray, dtype: Dtype) -> MlxArray {
    mlx_array_from_raw_or_panic(
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe { mlx_sys::mlx_astype(a.0, dtype.to_raw()) },
        "mlx_astype",
    )
}

pub fn zeros(shape: &[i32], dtype: Dtype) -> MlxArray {
    mlx_array_from_raw_or_panic(
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe { mlx_sys::mlx_zeros(shape.as_ptr(), shape.len(), dtype.to_raw()) },
        "mlx_zeros",
    )
}

pub fn take_axis(a: &MlxArray, indices: &MlxArray, axis: i32) -> MlxArray {
    mlx_array_from_raw_or_panic(
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe { mlx_sys::mlx_take_axis(a.0, indices.0, axis) },
        "mlx_take_axis",
    )
}

pub fn slice(a: &MlxArray, start: &[i32], stop: &[i32], strides: &[i32]) -> MlxArray {
    assert_eq!(
        start.len(),
        stop.len(),
        "mlx_slice start/stop rank mismatch"
    );
    assert_eq!(
        start.len(),
        strides.len(),
        "mlx_slice start/stride rank mismatch"
    );
    mlx_array_from_raw_or_panic(
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe {
            mlx_sys::mlx_slice(
                a.0,
                start.as_ptr(),
                stop.as_ptr(),
                strides.as_ptr(),
                start.len(),
            )
        },
        "mlx_slice",
    )
}

pub fn slice_update(src: &MlxArray, update: &MlxArray, start: &[i32], stop: &[i32]) -> MlxArray {
    assert_eq!(
        start.len(),
        stop.len(),
        "mlx_slice_update start/stop rank mismatch"
    );
    let strides = vec![1; start.len()];
    mlx_array_from_raw_or_panic(
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe {
            mlx_sys::mlx_slice_update(
                src.0,
                update.0,
                start.as_ptr(),
                stop.as_ptr(),
                strides.as_ptr(),
                start.len(),
            )
        },
        "mlx_slice_update",
    )
}

pub fn concatenate_axis(arrays: &[MlxArray], axis: i32) -> MlxArray {
    let raw: Vec<*mut mlx_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
    mlx_array_from_raw_or_panic(
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe { mlx_sys::mlx_concatenate_axis(raw.as_ptr().cast_mut(), raw.len(), axis) },
        "mlx_concatenate_axis",
    )
}

pub fn argmax(a: &MlxArray) -> MlxArray {
    // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
    mlx_array_from_raw_or_panic(unsafe { mlx_sys::mlx_argmax(a.0, false) }, "mlx_argmax")
}

pub fn argmax_axis(a: &MlxArray, axis: i32) -> MlxArray {
    mlx_array_from_raw_or_panic(
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe { mlx_sys::mlx_argmax_axis(a.0, axis, false) },
        "mlx_argmax_axis",
    )
}

pub fn dequantize(
    weight: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    group_size: i32,
    bits: i32,
) -> MlxArray {
    mlx_array_from_raw_or_panic(
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe { mlx_sys::mlx_dequantize(weight.0, scales.0, biases.0, group_size, bits) },
        "mlx_dequantize",
    )
}

pub fn quantized_matmul(
    x: &MlxArray,
    weight: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    transpose: bool,
    group_size: i32,
    bits: i32,
) -> MlxArray {
    mlx_array_from_raw_or_panic(
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe {
            mlx_sys::mlx_quantized_matmul(
                x.0, weight.0, scales.0, biases.0, transpose, group_size, bits,
            )
        },
        "mlx_quantized_matmul",
    )
}

pub fn eval(arrays: &[&MlxArray]) {
    let raw: Vec<*mut mlx_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
    // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
    unsafe {
        mlx_sys::mlx_eval(raw.as_ptr().cast_mut(), raw.len());
    }
    panic_if_mlx_error("mlx_eval");
}

pub fn async_eval(arrays: &[&MlxArray]) {
    let raw: Vec<*mut mlx_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
    // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
    unsafe {
        mlx_sys::mlx_async_eval(raw.as_ptr().cast_mut(), raw.len());
    }
    panic_if_mlx_error("mlx_async_eval");
}

pub fn set_wired_limit_bytes(limit: u64) -> u64 {
    // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
    let previous = unsafe { mlx_sys::mlx_set_wired_limit(limit as usize) as u64 };
    panic_if_mlx_error("mlx_set_wired_limit");
    previous
}

pub fn set_memory_limit_bytes(limit: u64) -> u64 {
    // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
    let previous = unsafe { mlx_sys::mlx_set_memory_limit(limit as usize) as u64 };
    panic_if_mlx_error("mlx_set_memory_limit");
    previous
}

pub fn set_cache_limit_bytes(limit: u64) -> u64 {
    // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
    let previous = unsafe { mlx_sys::mlx_set_cache_limit(limit as usize) as u64 };
    panic_if_mlx_error("mlx_set_cache_limit");
    previous
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocatorMemory {
    pub active_bytes: usize,
    pub peak_bytes: usize,
    pub cache_bytes: usize,
}

pub fn allocator_memory() -> AllocatorMemory {
    let stats = AllocatorMemory {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        active_bytes: unsafe { mlx_sys::mlx_get_active_memory() },
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        peak_bytes: unsafe { mlx_sys::mlx_get_peak_memory() },
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        cache_bytes: unsafe { mlx_sys::mlx_get_cache_memory() },
    };
    panic_if_mlx_error("mlx allocator memory stats");
    stats
}

pub fn clear_metal_cache() {
    // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
    unsafe {
        mlx_sys::mlx_metal_clear_cache();
    }
    panic_if_mlx_error("mlx_metal_clear_cache");
}

pub fn recommended_max_working_set_size_bytes() -> Option<usize> {
    // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
    let bytes = unsafe { mlx_sys::mlx_metal_recommended_max_working_set_size() };
    usize::try_from(bytes).ok().filter(|bytes| *bytes > 0)
}

pub fn load_safetensors(path: &str) -> anyhow::Result<std::collections::HashMap<String, MlxArray>> {
    let path = std::ffi::CString::new(path)?;
    let mut names: *mut *const i8 = std::ptr::null_mut();
    let mut arrays: *mut *mut mlx_sys::mlx_array = std::ptr::null_mut();
    // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
    let count = unsafe {
        mlx_sys::mlx_load_safetensors(
            path.as_ptr(),
            std::ptr::addr_of_mut!(names),
            std::ptr::addr_of_mut!(arrays),
        )
    };
    if count < 0 {
        return Err(check_mlx_error().unwrap_err());
    }

    let mut map = std::collections::HashMap::new();
    for i in 0..count as usize {
        // SAFETY: i < count and the bridge populated `names` with count valid NUL-terminated strings.
        let name = unsafe { CStr::from_ptr(*names.add(i)).to_string_lossy().to_string() };
        // SAFETY: i < count and `arrays[i]` is a valid handle; mlx_array_clone bumps its refcount.
        let cloned = unsafe { mlx_sys::mlx_array_clone(*arrays.add(i)) };
        map.insert(name, MlxArray(cloned));
    }
    if count > 0 {
        // SAFETY: mlx_sys FFI over valid owned handles and live caller buffers; failures are reported via rc/mlx_last_error checked after.
        unsafe {
            mlx_sys::mlx_free_loaded_tensors(names, arrays, count);
        }
    }
    Ok(map)
}
