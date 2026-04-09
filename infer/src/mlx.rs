//! Thin MLX wrapper — direct C++ bridge (no mlx-c), MLX v0.31.1.
//!
//! `MlxArray` wraps `*mut mlx_sys::mlx_array` (opaque pointer to `mlx::core::array*`).
//! No streams, no vector_array — the C++ bridge handles everything internally.

use std::os::raw::c_void;

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
    pub fn to_raw(self) -> i32 {
        match self {
            Dtype::Uint8 => mlx_sys::MLX_UINT8,
            Dtype::Uint32 => mlx_sys::MLX_UINT32,
            Dtype::Int32 => mlx_sys::MLX_INT32,
            Dtype::Float16 => mlx_sys::MLX_FLOAT16,
            Dtype::Float32 => mlx_sys::MLX_FLOAT32,
            Dtype::Bfloat16 => mlx_sys::MLX_BFLOAT16,
        }
    }
    pub fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            x if x == mlx_sys::MLX_UINT8 => Some(Dtype::Uint8),
            x if x == mlx_sys::MLX_UINT32 => Some(Dtype::Uint32),
            x if x == mlx_sys::MLX_INT32 => Some(Dtype::Int32),
            x if x == mlx_sys::MLX_FLOAT16 => Some(Dtype::Float16),
            x if x == mlx_sys::MLX_FLOAT32 => Some(Dtype::Float32),
            x if x == mlx_sys::MLX_BFLOAT16 => Some(Dtype::Bfloat16),
            _ => None,
        }
    }
}

pub struct MlxArray(*mut mlx_sys::mlx_array);

impl Drop for MlxArray {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                mlx_sys::mlx_array_free(self.0);
            }
        }
    }
}
impl Clone for MlxArray {
    fn clone(&self) -> Self {
        Self(unsafe { mlx_sys::mlx_array_clone(self.0) })
    }
}
impl std::fmt::Debug for MlxArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MlxArray({:?}, {:?})", self.shape(), self.dtype())
    }
}
unsafe impl Send for MlxArray {}
unsafe impl Sync for MlxArray {}

impl MlxArray {
    pub unsafe fn from_raw(raw: *mut mlx_sys::mlx_array) -> Self {
        Self(raw)
    }
    pub fn as_raw(&self) -> *mut mlx_sys::mlx_array {
        self.0
    }
    pub fn as_raw_mut(&mut self) -> *mut *mut mlx_sys::mlx_array {
        &mut self.0
    }
    pub fn into_raw(self) -> *mut mlx_sys::mlx_array {
        let r = self.0;
        std::mem::forget(self);
        r
    }

    pub unsafe fn from_raw_data(data: *const c_void, shape: &[i32], dtype: Dtype) -> Self {
        Self(mlx_sys::mlx_array_from_data(
            data,
            shape.as_ptr(),
            shape.len() as i32,
            dtype.to_raw(),
        ))
    }
    pub fn from_slice_i32(data: &[i32], shape: &[i32]) -> Self {
        unsafe { Self::from_raw_data(data.as_ptr().cast(), shape, Dtype::Int32) }
    }
    pub fn from_slice_f32(data: &[f32], shape: &[i32]) -> Self {
        unsafe { Self::from_raw_data(data.as_ptr().cast(), shape, Dtype::Float32) }
    }
    pub fn scalar_f32(val: f32) -> Self {
        Self(unsafe { mlx_sys::mlx_array_new_float32(val) })
    }
    pub fn scalar_i32(val: i32) -> Self {
        Self(unsafe { mlx_sys::mlx_array_new_int32(val) })
    }

    pub fn ndim(&self) -> usize {
        unsafe { mlx_sys::mlx_array_ndim(self.0) as usize }
    }
    pub fn shape(&self) -> &[i32] {
        unsafe {
            let ptr = mlx_sys::mlx_array_shape(self.0);
            let n = self.ndim();
            if n == 0 || ptr.is_null() {
                &[]
            } else {
                std::slice::from_raw_parts(ptr, n)
            }
        }
    }
    pub fn dtype(&self) -> Dtype {
        Dtype::from_raw(unsafe { mlx_sys::mlx_array_dtype(self.0) }).unwrap_or(Dtype::Float32)
    }
    pub fn item_i32(&self) -> i32 {
        unsafe { mlx_sys::mlx_array_item_int32(self.0) }
    }
    pub fn item_f32(&self) -> f32 {
        unsafe { mlx_sys::mlx_array_item_float32(self.0) }
    }
    pub fn as_slice_f32(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(
                mlx_sys::mlx_array_data_float32(self.0),
                mlx_sys::mlx_array_size(self.0),
            )
        }
    }
}

// ── Ops ──────────────────────────────────────────────────────────────────────

macro_rules! binary_op {
    ($name:ident, $cfn:ident) => {
        pub fn $name(a: &MlxArray, b: &MlxArray) -> MlxArray {
            MlxArray(unsafe { mlx_sys::$cfn(a.0, b.0) })
        }
    };
}
macro_rules! unary_op {
    ($name:ident, $cfn:ident) => {
        pub fn $name(a: &MlxArray) -> MlxArray {
            MlxArray(unsafe { mlx_sys::$cfn(a.0) })
        }
    };
}

binary_op!(add, mlx_add);
binary_op!(subtract, mlx_subtract);
binary_op!(multiply, mlx_multiply);
binary_op!(matmul, mlx_matmul);
binary_op!(greater, mlx_greater);
unary_op!(exp, mlx_exp);
unary_op!(log1p, mlx_log1p);
unary_op!(negative, mlx_negative);
unary_op!(sqrt, mlx_sqrt);
unary_op!(reciprocal, mlx_reciprocal);
unary_op!(sigmoid, mlx_sigmoid);

pub fn silu(a: &MlxArray) -> MlxArray {
    multiply(a, &sigmoid(a))
}

pub fn reshape(a: &MlxArray, s: &[i32]) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_reshape(a.0, s.as_ptr(), s.len()) })
}
pub fn transpose_all(a: &MlxArray) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_transpose(a.0) })
}
pub fn transpose_axes(a: &MlxArray, ax: &[i32]) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_transpose_axes(a.0, ax.as_ptr(), ax.len()) })
}
pub fn as_dtype(a: &MlxArray, d: Dtype) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_astype(a.0, d.to_raw()) })
}
pub fn zeros(s: &[i32], d: Dtype) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_zeros(s.as_ptr(), s.len(), d.to_raw()) })
}
pub fn take_axis(a: &MlxArray, idx: &MlxArray, ax: i32) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_take_axis(a.0, idx.0, ax) })
}
pub fn broadcast_to(a: &MlxArray, s: &[i32]) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_broadcast_to(a.0, s.as_ptr(), s.len()) })
}
pub fn expand_dims(a: &MlxArray, ax: i32) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_expand_dims(a.0, ax) })
}
pub fn sum_axis(a: &MlxArray, ax: i32, kd: bool) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_sum_axis(a.0, ax, kd) })
}
pub fn argmax(a: &MlxArray) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_argmax(a.0, false) })
}
pub fn where_(c: &MlxArray, a: &MlxArray, b: &MlxArray) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_where(c.0, a.0, b.0) })
}

pub fn concatenate_axis(arrays: &[MlxArray], axis: i32) -> MlxArray {
    let p: Vec<*mut mlx_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
    MlxArray(unsafe { mlx_sys::mlx_concatenate_axis(p.as_ptr() as *mut _, p.len(), axis) })
}
pub fn slice(a: &MlxArray, start: &[i32], stop: &[i32], strides: &[i32]) -> MlxArray {
    MlxArray(unsafe {
        mlx_sys::mlx_slice(
            a.0,
            start.as_ptr(),
            stop.as_ptr(),
            strides.as_ptr(),
            start.len(),
        )
    })
}
pub fn slice_update(a: &mut MlxArray, upd: &MlxArray, start: &[i32], stop: &[i32]) -> MlxArray {
    let st: Vec<i32> = vec![1; start.len()];
    MlxArray(unsafe {
        mlx_sys::mlx_slice_update(
            a.0,
            upd.0,
            start.as_ptr(),
            stop.as_ptr(),
            st.as_ptr(),
            start.len(),
        )
    })
}
pub fn quantized_matmul(
    x: &MlxArray,
    w: &MlxArray,
    s: &MlxArray,
    b: &MlxArray,
    tr: bool,
    gs: i32,
    bits: i32,
) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_quantized_matmul(x.0, w.0, s.0, b.0, tr, gs, bits) })
}
pub fn dequantize(w: &MlxArray, s: &MlxArray, b: &MlxArray, gs: i32, bits: i32) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_dequantize(w.0, s.0, b.0, gs, bits) })
}
pub fn contiguous(a: &MlxArray) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_contiguous(a.0) })
}
pub fn conv1d(inp: &MlxArray, w: &MlxArray, stride: i32, pad: i32, groups: i32) -> MlxArray {
    MlxArray(unsafe {
        mlx_sys::mlx_conv1d(inp.0, w.0, stride, pad, 1 /*dilation*/, groups)
    })
}
/// g = exp(-exp(A_log) * softplus(alpha + dt_bias))
/// Fused C++ implementation — replaces 10 individual FFI ops with 1.
pub fn compute_g(a_log: &MlxArray, alpha: &MlxArray, dt_bias: &MlxArray) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_compute_g(a_log.0, alpha.0, dt_bias.0) })
}

/// Full GDR layer forward in C++ — one FFI call replaces ~40 individual ops.
/// Returns the output tensor. Updates conv_state and gdr_state in place.
#[allow(clippy::too_many_arguments)]
pub fn gdr_layer_forward(
    x: &MlxArray,
    qkvz_w: &MlxArray,
    qkvz_s: &MlxArray,
    qkvz_b: &MlxArray,
    qkvz_gs: i32,
    qkvz_bits: i32,
    qkv_split: i32,
    z_split: i32,
    ba_w: &MlxArray,
    ba_s: &MlxArray,
    ba_b: &MlxArray,
    ba_gs: i32,
    ba_bits: i32,
    ba_num_heads: i32,
    conv1d_w: &MlxArray,
    conv_state: &mut MlxArray,
    conv_kernel: i32,
    a_log: &MlxArray,
    dt_bias: &MlxArray,
    norm_w: &MlxArray,
    rms_eps: f32,
    out_w: &MlxArray,
    out_s: &MlxArray,
    out_b: &MlxArray,
    out_gs: i32,
    out_bits: i32,
    num_key_heads: i32,
    key_dim: i32,
    num_value_heads: i32,
    value_dim: i32,
    q_scale: f32,
    k_scale: f32,
    gdr_state: &mut MlxArray,
    metal_kernel: *mut std::ffi::c_void,
    use_metal_kernel: bool,
) -> MlxArray {
    unsafe {
        let mut result: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        mlx_sys::mlx_gdr_layer_forward(
            x.0,
            qkvz_w.0,
            qkvz_s.0,
            qkvz_b.0,
            qkvz_gs,
            qkvz_bits,
            qkv_split,
            z_split,
            ba_w.0,
            ba_s.0,
            ba_b.0,
            ba_gs,
            ba_bits,
            ba_num_heads,
            conv1d_w.0,
            &mut conv_state.0 as *mut _,
            conv_kernel,
            a_log.0,
            dt_bias.0,
            norm_w.0,
            rms_eps,
            out_w.0,
            out_s.0,
            out_b.0,
            out_gs,
            out_bits,
            num_key_heads,
            key_dim,
            num_value_heads,
            value_dim,
            q_scale,
            k_scale,
            &mut gdr_state.0 as *mut _,
            metal_kernel,
            use_metal_kernel as i32,
            &mut result,
        );
        MlxArray(result)
    }
}

pub fn rms_norm(x: &MlxArray, w: &MlxArray, eps: f32) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_fast_rms_norm(x.0, w.0, eps) })
}
pub fn rms_norm_no_weight(x: &MlxArray, eps: f32) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_fast_rms_norm(x.0, std::ptr::null_mut(), eps) })
}
pub fn rope(x: &MlxArray, dims: i32, trad: bool, base: f32, scale: f32, off: i32) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_fast_rope(x.0, dims, trad, base, scale, off) })
}
pub fn scaled_dot_product_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: Option<&str>,
) -> MlxArray {
    let m = std::ffi::CString::new(mask.unwrap_or("")).unwrap();
    MlxArray(unsafe { mlx_sys::mlx_fast_sdpa(q.0, k.0, v.0, scale, m.as_ptr()) })
}
pub fn categorical(logits: &MlxArray) -> MlxArray {
    MlxArray(unsafe { mlx_sys::mlx_random_categorical(logits.0, -1) })
}

pub fn eval(arrays: &[&MlxArray]) {
    let p: Vec<*mut mlx_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
    unsafe {
        mlx_sys::mlx_eval(p.as_ptr() as *mut _, p.len());
    }
}
pub fn async_eval(arrays: &[&MlxArray]) {
    let p: Vec<*mut mlx_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
    unsafe {
        mlx_sys::mlx_async_eval(p.as_ptr() as *mut _, p.len());
    }
}

pub struct MetalKernel(*mut std::ffi::c_void);
impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_metal_kernel_free(self.0);
        }
    }
}
impl MetalKernel {
    /// Return the raw opaque pointer for passing to C++ functions.
    pub fn as_raw(&self) -> *mut std::ffi::c_void {
        self.0
    }
}
unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl MetalKernel {
    pub fn new(name: &str, input_names: &[&str], output_names: &[&str], source: &str) -> Self {
        use std::ffi::CString;
        let n = CString::new(name).unwrap();
        let s = CString::new(source).unwrap();
        let h = CString::new("").unwrap();
        let ic: Vec<CString> = input_names
            .iter()
            .map(|x| CString::new(*x).unwrap())
            .collect();
        let ip: Vec<*const i8> = ic.iter().map(|c| c.as_ptr()).collect();
        let oc: Vec<CString> = output_names
            .iter()
            .map(|x| CString::new(*x).unwrap())
            .collect();
        let op: Vec<*const i8> = oc.iter().map(|c| c.as_ptr()).collect();
        Self(unsafe {
            mlx_sys::mlx_metal_kernel_new(
                n.as_ptr(),
                ip.as_ptr(),
                ip.len(),
                op.as_ptr(),
                op.len(),
                s.as_ptr(),
                h.as_ptr(),
            )
        })
    }

    pub fn apply(
        &self,
        inputs: &[&MlxArray],
        grid: [i32; 3],
        tg: [i32; 3],
        out_shapes: &[&[i32]],
        out_dtypes: &[Dtype],
        int_tmpl: &[(&str, i32)],
        dtype_tmpl: &[(&str, Dtype)],
    ) -> Vec<MlxArray> {
        use std::ffi::CString;
        let ip: Vec<*mut mlx_sys::mlx_array> = inputs.iter().map(|a| a.0).collect();
        let no = out_shapes.len();
        let mut op: Vec<*mut mlx_sys::mlx_array> = vec![std::ptr::null_mut(); no];
        let mut sd: Vec<i32> = Vec::new();
        let mut dd: Vec<i32> = Vec::new();
        for s in out_shapes {
            dd.push(s.len() as i32);
            sd.extend_from_slice(s);
        }
        let dt: Vec<i32> = out_dtypes.iter().map(|d| d.to_raw()).collect();
        let inc: Vec<CString> = int_tmpl
            .iter()
            .map(|(n, _)| CString::new(*n).unwrap())
            .collect();
        let dnc: Vec<CString> = dtype_tmpl
            .iter()
            .map(|(n, _)| CString::new(*n).unwrap())
            .collect();
        let mut np: Vec<*const i8> = inc.iter().map(|c| c.as_ptr()).collect();
        np.extend(dnc.iter().map(|c| c.as_ptr()));
        let iv: Vec<i32> = int_tmpl.iter().map(|(_, v)| *v).collect();
        let dv: Vec<i32> = dtype_tmpl.iter().map(|(_, d)| d.to_raw()).collect();
        unsafe {
            mlx_sys::mlx_metal_kernel_apply(
                self.0,
                ip.as_ptr() as *mut _,
                ip.len(),
                op.as_mut_ptr(),
                no,
                grid.as_ptr(),
                tg.as_ptr(),
                sd.as_ptr(),
                dd.as_ptr(),
                dt.as_ptr(),
                np.as_ptr(),
                iv.as_ptr(),
                dv.as_ptr(),
                iv.len(),
                dv.len(),
            );
        }
        op.into_iter().map(|p| MlxArray(p)).collect()
    }
}

pub fn load_safetensors(path: &str) -> std::collections::HashMap<String, MlxArray> {
    let pc = std::ffi::CString::new(path).unwrap();
    let mut names: *mut *const i8 = std::ptr::null_mut();
    let mut arrays: *mut *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let count = unsafe { mlx_sys::mlx_load_safetensors(pc.as_ptr(), &mut names, &mut arrays) };
    let mut map = std::collections::HashMap::new();
    for i in 0..count as usize {
        let name = unsafe {
            std::ffi::CStr::from_ptr(*names.add(i))
                .to_string_lossy()
                .to_string()
        };
        // Clone (refcount++) so Rust owns a separate reference
        let cloned = unsafe { mlx_sys::mlx_array_clone(*arrays.add(i)) };
        map.insert(name, MlxArray(cloned));
    }
    if count > 0 {
        // Free C++ allocations: name strings, name array, array objects, array container.
        // Our cloned arrays survive because they hold their own shared_ptr reference.
        unsafe {
            mlx_sys::mlx_free_loaded_tensors(names as *mut _, arrays, count);
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lifecycle() {
        let a = MlxArray::from_slice_f32(&[1.0, 2.0, 3.0], &[3]);
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.dtype(), Dtype::Float32);
    }
    #[test]
    fn add_basic() {
        let c = add(
            &MlxArray::from_slice_f32(&[1.0, 2.0], &[2]),
            &MlxArray::from_slice_f32(&[3.0, 4.0], &[2]),
        );
        eval(&[&c]);
        assert!((c.as_slice_f32()[0] - 4.0).abs() < 1e-6);
    }
    #[test]
    fn matmul_basic() {
        let c = matmul(
            &MlxArray::from_slice_f32(&[1.0, 2.0], &[1, 2]),
            &MlxArray::from_slice_f32(&[1.0, 2.0], &[2, 1]),
        );
        eval(&[&c]);
        assert!((c.as_slice_f32()[0] - 5.0).abs() < 1e-6);
    }
}

pub fn compile_clear_cache() {
    // No-op with new mlx-sys — compile cache management not yet ported
}
