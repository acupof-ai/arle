//! Thin MLX wrapper — direct C++ bridge (no mlx-c), MLX v0.31.1.
//!
//! `MlxArray` wraps `*mut mlx_sys::mlx_array` (opaque pointer to `mlx::core::array*`).
//! No streams, no vector_array — the C++ bridge handles everything internally.

use std::os::raw::c_void;

/// Check the MLX C++ bridge for a pending error. Returns `Err` if an MLX
/// exception was caught by a try/catch wrapper, `Ok(())` otherwise.
pub fn check_mlx_error() -> anyhow::Result<()> {
    unsafe {
        let ptr = mlx_sys::mlx_last_error();
        if ptr.is_null() {
            Ok(())
        } else {
            let msg = std::ffi::CStr::from_ptr(ptr).to_string_lossy();
            Err(anyhow::anyhow!("MLX error: {msg}"))
        }
    }
}

fn mlx_error_message() -> Option<String> {
    unsafe {
        let ptr = mlx_sys::mlx_last_error();
        (!ptr.is_null()).then(|| std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned())
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
        mlx_array_from_raw_or_panic(
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
unsafe impl Send for MlxArray {}

impl MlxArray {
    /// # Safety
    /// `raw` must be a valid owned MLX array handle returned by the bridge.
    pub unsafe fn from_raw(raw: *mut mlx_sys::mlx_array) -> Self {
        mlx_array_from_raw_or_panic(raw, "MlxArray::from_raw")
    }
    /// Like `from_raw` but returns `Result` instead of panicking on null.
    ///
    /// # Safety
    /// `raw` must be a valid owned MLX array handle returned by the bridge,
    /// or null with a bridge-side error already recorded for retrieval via
    /// `check_mlx_error()`.
    pub unsafe fn from_raw_checked(raw: *mut mlx_sys::mlx_array) -> anyhow::Result<Self> {
        if raw.is_null() {
            check_mlx_error()?;
            anyhow::bail!("MLX returned null array");
        }
        Ok(Self(raw))
    }
    pub fn as_raw(&self) -> *mut mlx_sys::mlx_array {
        self.0
    }
    pub fn as_raw_mut(&mut self) -> *mut *mut mlx_sys::mlx_array {
        std::ptr::addr_of_mut!(self.0)
    }
    pub fn into_raw(self) -> *mut mlx_sys::mlx_array {
        let r = self.0;
        std::mem::forget(self);
        r
    }

    /// # Safety
    /// `data` must remain valid for MLX to read for the duration required by the bridge.
    pub unsafe fn from_raw_data(data: *const c_void, shape: &[i32], dtype: Dtype) -> Self {
        mlx_array_from_raw_or_panic(
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
        unsafe { Self::from_raw_data(data.as_ptr().cast(), shape, Dtype::Int32) }
    }
    pub fn from_slice_f32(data: &[f32], shape: &[i32]) -> Self {
        unsafe { Self::from_raw_data(data.as_ptr().cast(), shape, Dtype::Float32) }
    }
    pub fn scalar_f32(val: f32) -> Self {
        mlx_array_from_raw_or_panic(
            unsafe { mlx_sys::mlx_array_new_float32(val) },
            "mlx_array_new_float32",
        )
    }
    pub fn scalar_i32(val: i32) -> Self {
        mlx_array_from_raw_or_panic(
            unsafe { mlx_sys::mlx_array_new_int32(val) },
            "mlx_array_new_int32",
        )
    }

    pub fn ndim(&self) -> usize {
        let ndim = unsafe { mlx_sys::mlx_array_ndim(self.0) as usize };
        panic_if_mlx_error("mlx_array_ndim");
        ndim
    }
    pub fn shape(&self) -> &[i32] {
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
        let raw = unsafe { mlx_sys::mlx_array_dtype(self.0) };
        panic_if_mlx_error("mlx_array_dtype");
        Dtype::from_raw(raw)
            .unwrap_or_else(|| panic!("mlx_array_dtype returned unknown dtype {raw}"))
    }
    pub fn item_i32(&self) -> i32 {
        let value = unsafe { mlx_sys::mlx_array_item_int32(self.0) };
        panic_if_mlx_error("mlx_array_item_int32");
        value
    }
    pub fn item_f32(&self) -> f32 {
        let value = unsafe { mlx_sys::mlx_array_item_float32(self.0) };
        panic_if_mlx_error("mlx_array_item_float32");
        value
    }
    pub fn as_slice_f32(&self) -> &[f32] {
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
    pub fn as_slice_i32(&self) -> Vec<i32> {
        unsafe {
            let ptr = mlx_sys::mlx_array_data_int32(self.0);
            let len = mlx_sys::mlx_array_size(self.0);
            if ptr.is_null() && len > 0 {
                panic_if_mlx_error("mlx_array_data_int32");
                panic!("mlx_array_data_int32 returned null for non-empty array");
            }
            panic_if_mlx_error("mlx_array_data_int32");
            std::slice::from_raw_parts(ptr, len).to_vec()
        }
    }
}

// ── Ops ──────────────────────────────────────────────────────────────────────

macro_rules! binary_op {
    ($name:ident, $cfn:ident) => {
        pub fn $name(a: &MlxArray, b: &MlxArray) -> MlxArray {
            mlx_array_from_raw_or_panic(unsafe { mlx_sys::$cfn(a.0, b.0) }, stringify!($cfn))
        }
    };
}
macro_rules! unary_op {
    ($name:ident, $cfn:ident) => {
        pub fn $name(a: &MlxArray) -> MlxArray {
            mlx_array_from_raw_or_panic(unsafe { mlx_sys::$cfn(a.0) }, stringify!($cfn))
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
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_reshape(a.0, s.as_ptr(), s.len()) },
        "mlx_reshape",
    )
}
pub fn transpose_all(a: &MlxArray) -> MlxArray {
    mlx_array_from_raw_or_panic(unsafe { mlx_sys::mlx_transpose(a.0) }, "mlx_transpose")
}
pub fn transpose_axes(a: &MlxArray, ax: &[i32]) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_transpose_axes(a.0, ax.as_ptr(), ax.len()) },
        "mlx_transpose_axes",
    )
}
pub fn as_dtype(a: &MlxArray, d: Dtype) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_astype(a.0, d.to_raw()) },
        "mlx_astype",
    )
}
pub fn zeros(s: &[i32], d: Dtype) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_zeros(s.as_ptr(), s.len(), d.to_raw()) },
        "mlx_zeros",
    )
}
pub fn take_axis(a: &MlxArray, idx: &MlxArray, ax: i32) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_take_axis(a.0, idx.0, ax) },
        "mlx_take_axis",
    )
}
pub fn broadcast_to(a: &MlxArray, s: &[i32]) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_broadcast_to(a.0, s.as_ptr(), s.len()) },
        "mlx_broadcast_to",
    )
}
pub fn expand_dims(a: &MlxArray, ax: i32) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_expand_dims(a.0, ax) },
        "mlx_expand_dims",
    )
}
pub fn sum_axis(a: &MlxArray, ax: i32, kd: bool) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_sum_axis(a.0, ax, kd) },
        "mlx_sum_axis",
    )
}
pub fn argmax(a: &MlxArray) -> MlxArray {
    mlx_array_from_raw_or_panic(unsafe { mlx_sys::mlx_argmax(a.0, false) }, "mlx_argmax")
}
pub fn argmax_axis(a: &MlxArray, axis: i32) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_argmax_axis(a.0, axis, false) },
        "mlx_argmax_axis",
    )
}
pub fn where_(c: &MlxArray, a: &MlxArray, b: &MlxArray) -> MlxArray {
    mlx_array_from_raw_or_panic(unsafe { mlx_sys::mlx_where(c.0, a.0, b.0) }, "mlx_where")
}

pub fn prefix_match_len_i32(lhs: &MlxArray, rhs: &MlxArray) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_prefix_match_len_i32(lhs.0, rhs.0) },
        "mlx_prefix_match_len_i32",
    )
}

pub fn concatenate_axis(arrays: &[MlxArray], axis: i32) -> MlxArray {
    let p: Vec<*mut mlx_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_concatenate_axis(p.as_ptr().cast_mut(), p.len(), axis) },
        "mlx_concatenate_axis",
    )
}
pub fn slice(a: &MlxArray, start: &[i32], stop: &[i32], strides: &[i32]) -> MlxArray {
    mlx_array_from_raw_or_panic(
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
pub fn slice_update(a: &mut MlxArray, upd: &MlxArray, start: &[i32], stop: &[i32]) -> MlxArray {
    let st: Vec<i32> = vec![1; start.len()];
    mlx_array_from_raw_or_panic(
        unsafe {
            mlx_sys::mlx_slice_update(
                a.0,
                upd.0,
                start.as_ptr(),
                stop.as_ptr(),
                st.as_ptr(),
                start.len(),
            )
        },
        "mlx_slice_update",
    )
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
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_quantized_matmul(x.0, w.0, s.0, b.0, tr, gs, bits) },
        "mlx_quantized_matmul",
    )
}
pub fn dequantize(w: &MlxArray, s: &MlxArray, b: &MlxArray, gs: i32, bits: i32) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_dequantize(w.0, s.0, b.0, gs, bits) },
        "mlx_dequantize",
    )
}
pub fn contiguous(a: &MlxArray) -> MlxArray {
    mlx_array_from_raw_or_panic(unsafe { mlx_sys::mlx_contiguous(a.0) }, "mlx_contiguous")
}
pub fn conv1d(inp: &MlxArray, w: &MlxArray, stride: i32, pad: i32, groups: i32) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe {
            mlx_sys::mlx_conv1d(inp.0, w.0, stride, pad, 1 /*dilation*/, groups)
        },
        "mlx_conv1d",
    )
}
/// g = exp(-exp(A_log) * softplus(alpha + dt_bias))
/// Fused C++ implementation — replaces 10 individual FFI ops with 1.
pub fn compute_g(a_log: &MlxArray, alpha: &MlxArray, dt_bias: &MlxArray) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_compute_g(a_log.0, alpha.0, dt_bias.0) },
        "mlx_compute_g",
    )
}

/// Full GDR layer forward in C++ — one FFI call replaces ~40 individual ops.
/// Returns the output tensor. Updates conv_state and gdr_state in place.
///
/// `metal_kernel` is a raw Metal kernel pointer that must be valid for the
/// duration of the call when `use_metal_kernel` is true; callers are
/// responsible for its lifetime.
#[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
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
            std::ptr::addr_of_mut!(conv_state.0),
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
            std::ptr::addr_of_mut!(gdr_state.0),
            metal_kernel,
            use_metal_kernel as i32,
            std::ptr::addr_of_mut!(result),
        );
        mlx_array_from_raw_or_panic(result, "mlx_gdr_layer_forward")
    }
}

pub fn rms_norm(x: &MlxArray, w: &MlxArray, eps: f32) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_fast_rms_norm(x.0, w.0, eps) },
        "mlx_fast_rms_norm",
    )
}
pub fn rms_norm_no_weight(x: &MlxArray, eps: f32) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_fast_rms_norm(x.0, std::ptr::null_mut(), eps) },
        "mlx_fast_rms_norm",
    )
}
pub fn rope(x: &MlxArray, dims: i32, trad: bool, base: f32, scale: f32, off: i32) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_fast_rope(x.0, dims, trad, base, scale, off) },
        "mlx_fast_rope",
    )
}
pub fn rope_dynamic(
    x: &MlxArray,
    dims: i32,
    trad: bool,
    base: f32,
    scale: f32,
    off: &MlxArray,
) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_fast_rope_dynamic(x.0, dims, trad, base, scale, off.0) },
        "mlx_fast_rope_dynamic",
    )
}
pub fn build_varlen_decode_mask(left_padding: &[i32], batch_cache_len: i32) -> MlxArray {
    debug_assert!(!left_padding.is_empty());
    debug_assert!(batch_cache_len >= 0);
    debug_assert!(i32::try_from(left_padding.len()).is_ok());
    debug_assert!(
        left_padding
            .iter()
            .all(|&padding| (0..=batch_cache_len).contains(&padding))
    );

    let batch = left_padding.len() as i32;
    let key_len = batch_cache_len + 1;
    let cols_data: Vec<f32> = (0..key_len).map(|col| col as f32).collect();
    let pad_data: Vec<f32> = left_padding.iter().map(|&padding| padding as f32).collect();

    let cols = MlxArray::from_slice_f32(&cols_data, &[1, key_len]);
    let cols = broadcast_to(&cols, &[batch, key_len]);
    let pad = MlxArray::from_slice_f32(&pad_data, &[batch, 1]);
    let pad = broadcast_to(&pad, &[batch, key_len]);
    let cond = greater(&pad, &cols);

    let neg_inf = MlxArray::scalar_f32(f32::NEG_INFINITY);
    let neg_inf = broadcast_to(&neg_inf, &[batch, key_len]);
    let zero = zeros(&[batch, key_len], Dtype::Float32);
    let mask = where_(&cond, &neg_inf, &zero);
    let mask = expand_dims(&expand_dims(&mask, 1), 1);
    // MLX >= 0.32 requires the additive mask dtype to match/promote to the
    // SDPA output dtype. Q/K/V are bf16 here, so f32 no longer auto-promotes
    // — cast explicitly.
    let mask = as_dtype(&mask, Dtype::Bfloat16);
    eval(&[&mask]);
    mask
}

/// Build the additive `[B, 1, block_size, key_len]` SDPA mask for a packed
/// DFlash verify forward where each row may have a different left-pad and the
/// `block_size` verify positions extend the packed cache.
///
/// `key_len = batch_cache_len + block_size`. Cell `[b, 0, q, k]` = `-inf` iff
///   - `k < left_padding[b]` (left-pad column), OR
///   - `(k - batch_cache_len) > q` (causal mask within the verify block —
///     position `q` may attend to verify-cols `0..=q` but not `q+1..`).
///
/// The two conditions compose via additive sum: `-inf + 0 = -inf`,
/// `-inf + -inf = -inf`, `0 + 0 = 0`. Cast to bf16 to match Q/K/V dtype
/// (mirrors `build_varlen_decode_mask`).
pub fn build_varlen_verify_mask(
    left_padding: &[i32],
    block_size: i32,
    batch_cache_len: i32,
) -> MlxArray {
    debug_assert!(!left_padding.is_empty());
    debug_assert!(block_size > 0);
    debug_assert!(batch_cache_len >= 0);
    debug_assert!(i32::try_from(left_padding.len()).is_ok());
    debug_assert!(
        left_padding
            .iter()
            .all(|&padding| (0..=batch_cache_len).contains(&padding))
    );

    let batch = left_padding.len() as i32;
    let key_len = batch_cache_len + block_size;

    // Broadcasted key-column index `[B, 1, block_size, key_len]`.
    let cols_data: Vec<f32> = (0..key_len).map(|col| col as f32).collect();
    let cols = MlxArray::from_slice_f32(&cols_data, &[1, 1, 1, key_len]);
    let cols_b = broadcast_to(&cols, &[batch, 1, block_size, key_len]);

    // Per-row left-pad threshold `[B, 1, block_size, key_len]`.
    let pad_data: Vec<f32> = left_padding.iter().map(|&padding| padding as f32).collect();
    let pad = MlxArray::from_slice_f32(&pad_data, &[batch, 1, 1, 1]);
    let pad_b = broadcast_to(&pad, &[batch, 1, block_size, key_len]);
    let cond_pad = greater(&pad_b, &cols_b);

    // `delta = cols - batch_cache_len`; causal cond is `delta > q_idx`.
    // For `cols < batch_cache_len`, `delta < 0` while `q_idx >= 0`, so the
    // condition is automatically false — we don't need a separate
    // `cols >= batch_cache_len` clause.
    let bcl = MlxArray::from_slice_f32(&[batch_cache_len as f32], &[1, 1, 1, 1]);
    let bcl_b = broadcast_to(&bcl, &[batch, 1, block_size, key_len]);
    let delta = subtract(&cols_b, &bcl_b);

    let q_data: Vec<f32> = (0..block_size).map(|q| q as f32).collect();
    let q_idx = MlxArray::from_slice_f32(&q_data, &[1, 1, block_size, 1]);
    let q_idx_b = broadcast_to(&q_idx, &[batch, 1, block_size, key_len]);
    let cond_causal = greater(&delta, &q_idx_b);

    // Additive composition: `-inf + 0 = -inf`, `-inf + -inf = -inf`, `0 = 0`.
    let neg_inf = MlxArray::scalar_f32(f32::NEG_INFINITY);
    let neg_inf_b = broadcast_to(&neg_inf, &[batch, 1, block_size, key_len]);
    let zero_b = zeros(&[batch, 1, block_size, key_len], Dtype::Float32);
    let pad_term = where_(&cond_pad, &neg_inf_b, &zero_b);
    let causal_term = where_(&cond_causal, &neg_inf_b, &zero_b);
    let mask = add(&pad_term, &causal_term);
    let mask = as_dtype(&mask, Dtype::Bfloat16);
    eval(&[&mask]);
    mask
}
pub fn scaled_dot_product_attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: Option<&str>,
) -> MlxArray {
    let m = std::ffi::CString::new(mask.unwrap_or("")).unwrap();
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_fast_sdpa(q.0, k.0, v.0, scale, m.as_ptr()) },
        "mlx_fast_sdpa",
    )
}
pub fn scaled_dot_product_attention_masked(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: &MlxArray,
) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_fast_sdpa_masked(q.0, k.0, v.0, scale, mask.0) },
        "mlx_fast_sdpa_masked",
    )
}
pub fn categorical(logits: &MlxArray) -> MlxArray {
    mlx_array_from_raw_or_panic(
        unsafe { mlx_sys::mlx_random_categorical(logits.0, -1) },
        "mlx_random_categorical",
    )
}

pub fn eval(arrays: &[&MlxArray]) {
    let p: Vec<*mut mlx_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
    unsafe {
        mlx_sys::mlx_eval(p.as_ptr().cast_mut(), p.len());
    }
    panic_if_mlx_error("mlx_eval");
}
pub fn async_eval(arrays: &[&MlxArray]) {
    let p: Vec<*mut mlx_sys::mlx_array> = arrays.iter().map(|a| a.0).collect();
    unsafe {
        mlx_sys::mlx_async_eval(p.as_ptr().cast_mut(), p.len());
    }
    panic_if_mlx_error("mlx_async_eval");
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
// SAFETY: `MetalKernel` is an immutable handle to a compiled MLX Metal kernel.
// We construct it once, store it in a process-wide `LazyLock`, and only call
// methods that treat the handle as read-only. Per-dispatch mutable state lives
// in the input/output arrays passed to MLX, not in this wrapper.
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
        let raw = unsafe {
            mlx_sys::mlx_metal_kernel_new(
                n.as_ptr(),
                ip.as_ptr(),
                ip.len(),
                op.as_ptr(),
                op.len(),
                s.as_ptr(),
                h.as_ptr(),
            )
        };
        panic_if_mlx_error("mlx_metal_kernel_new");
        assert!(!raw.is_null(), "mlx_metal_kernel_new returned null");
        Self(raw)
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
                ip.as_ptr().cast_mut(),
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
        panic_if_mlx_error("mlx_metal_kernel_apply");
        op.into_iter()
            .map(|p| mlx_array_from_raw_or_panic(p, "mlx_metal_kernel_apply"))
            .collect()
    }
}

pub fn load_safetensors(path: &str) -> anyhow::Result<std::collections::HashMap<String, MlxArray>> {
    let pc = std::ffi::CString::new(path).unwrap();
    let mut names: *mut *const i8 = std::ptr::null_mut();
    let mut arrays: *mut *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let count = unsafe {
        mlx_sys::mlx_load_safetensors(
            pc.as_ptr(),
            std::ptr::addr_of_mut!(names),
            std::ptr::addr_of_mut!(arrays),
        )
    };
    if count < 0 {
        // C++ threw an exception — check the error string
        return Err(check_mlx_error().unwrap_err());
    }
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
            mlx_sys::mlx_free_loaded_tensors(names, arrays, count);
        }
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::metal_test_guard;

    #[test]
    fn lifecycle() {
        let _guard = metal_test_guard();
        let a = MlxArray::from_slice_f32(&[1.0, 2.0, 3.0], &[3]);
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.dtype(), Dtype::Float32);
    }
    #[test]
    fn add_basic() {
        let _guard = metal_test_guard();
        let c = add(
            &MlxArray::from_slice_f32(&[1.0, 2.0], &[2]),
            &MlxArray::from_slice_f32(&[3.0, 4.0], &[2]),
        );
        eval(&[&c]);
        assert!((c.as_slice_f32()[0] - 4.0).abs() < 1e-6);
    }
    #[test]
    fn matmul_basic() {
        let _guard = metal_test_guard();
        let c = matmul(
            &MlxArray::from_slice_f32(&[1.0, 2.0], &[1, 2]),
            &MlxArray::from_slice_f32(&[1.0, 2.0], &[2, 1]),
        );
        eval(&[&c]);
        assert!((c.as_slice_f32()[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn build_varlen_decode_mask_marks_left_padding() {
        let _guard = metal_test_guard();
        let mask = build_varlen_decode_mask(&[2, 0], 3);
        assert_eq!(mask.shape(), &[2, 1, 1, 4]);

        // mask is bf16 now (see build_varlen_decode_mask); cast back to f32 for inspection.
        let mask_f32 = as_dtype(&mask, Dtype::Float32);
        eval(&[&mask_f32]);
        let values = mask_f32.as_slice_f32();
        assert!(values[0].is_infinite() && values[0].is_sign_negative());
        assert!(values[1].is_infinite() && values[1].is_sign_negative());
        assert_eq!(values[2], 0.0);
        assert_eq!(values[4], 0.0);
    }

    #[test]
    fn build_varlen_verify_mask_b2_matches_reference() {
        let _guard = metal_test_guard();

        let left_padding = [0_i32, 2];
        let block_size = 4_i32;
        let batch_cache_len = 5_i32;
        let key_len = batch_cache_len + block_size; // 9
        let batch = left_padding.len() as i32;

        let mask = build_varlen_verify_mask(&left_padding, block_size, batch_cache_len);
        assert_eq!(mask.shape(), &[batch, 1, block_size, key_len]);

        let mask_f32 = as_dtype(&mask, Dtype::Float32);
        eval(&[&mask_f32]);
        let actual = mask_f32.as_slice_f32();

        // Build expected mask in row-major layout `[B, 1, block_size, key_len]`.
        let mut expected: Vec<f32> = Vec::with_capacity(actual.len());
        for b in 0..batch {
            let pad = left_padding[b as usize];
            for q in 0..block_size {
                for k in 0..key_len {
                    let pad_block = k < pad;
                    let causal_future = k >= batch_cache_len && (k - batch_cache_len) > q;
                    let masked = pad_block || causal_future;
                    expected.push(if masked { f32::NEG_INFINITY } else { 0.0 });
                }
            }
        }

        assert_eq!(
            actual.len(),
            expected.len(),
            "mask len mismatch: got {}, expected {}",
            actual.len(),
            expected.len()
        );

        let mut max_abs_delta = 0.0_f32;
        for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
            // Treat both being `-inf` as exact match.
            if lhs.is_infinite()
                && rhs.is_infinite()
                && lhs.is_sign_negative() == rhs.is_sign_negative()
            {
                continue;
            }
            let delta = (lhs - rhs).abs();
            assert_eq!(
                delta, 0.0,
                "mismatch at flat idx {idx}: got {lhs}, expected {rhs}"
            );
            max_abs_delta = max_abs_delta.max(delta);
        }
        assert_eq!(max_abs_delta, 0.0);
    }

    #[test]
    fn prefix_match_len_i32_counts_only_the_matching_prefix() {
        let _guard = metal_test_guard();
        let lhs = MlxArray::from_slice_i32(&[11, 12, 13, 14], &[4]);
        let rhs = MlxArray::from_slice_i32(&[11, 12, 99, 14], &[4]);
        let matched = prefix_match_len_i32(&lhs, &rhs);
        eval(&[&matched]);
        assert_eq!(matched.item_i32(), 2);
    }

    // MLX 0.31.1: `fast::rope(..., int offset)` on a `[B, H, S=1, D]` tensor
    // with `B > 1` silently zeroes out batch rows > 0. The array-offset
    // overload (`rope_dynamic`) works correctly. This test pins the
    // workaround so a future MLX upgrade (or an accidental swap back to the
    // scalar path) can't silently regress Qwen3/Qwen3.5 batched decode.
    //
    // Repro of the raw bug is skipped — it depends on MLX lazy-eval global
    // state which makes it flaky under `--test-threads`. See
    // `docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md`.

    #[test]
    fn rope_dynamic_works_on_b_gt_1_s_eq_1_and_matches_per_row_reference() {
        // The actual workaround: feed an int32[B] offsets array, get
        // correct per-row rotation. Verifies row 0 == scalar B=1 at its
        // offset AND row 1 == scalar B=1 at its offset.
        let _guard = metal_test_guard();
        let values: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let x = MlxArray::from_slice_f32(&values, &[2, 3, 1, 4]);

        // Per-row offsets: row 0 @ pos 5, row 1 @ pos 3.
        let offsets = MlxArray::from_slice_i32(&[5, 3], &[2]);
        let batched = rope_dynamic(&x, 4, false, 10000.0, 1.0, &offsets);
        eval(&[&batched]);

        // Per-row references via B=1 scalar rope (which works correctly).
        let row0_values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let row0 = MlxArray::from_slice_f32(&row0_values, &[1, 3, 1, 4]);
        let row0_ref = rope(&row0, 4, false, 10000.0, 1.0, 5);

        let row1_values: Vec<f32> = (12..24).map(|v| v as f32).collect();
        let row1 = MlxArray::from_slice_f32(&row1_values, &[1, 3, 1, 4]);
        let row1_ref = rope(&row1, 4, false, 10000.0, 1.0, 3);

        eval(&[&row0_ref, &row1_ref]);

        let batched_flat = batched.as_slice_f32();
        let row0_ref_flat = row0_ref.as_slice_f32();
        let row1_ref_flat = row1_ref.as_slice_f32();

        for (i, (lhs, rhs)) in batched_flat[0..12]
            .iter()
            .zip(row0_ref_flat.iter())
            .enumerate()
        {
            assert!((lhs - rhs).abs() < 1e-5, "row 0 index {i}: {lhs} != {rhs}");
        }
        for (i, (lhs, rhs)) in batched_flat[12..24]
            .iter()
            .zip(row1_ref_flat.iter())
            .enumerate()
        {
            assert!((lhs - rhs).abs() < 1e-5, "row 1 index {i}: {lhs} != {rhs}");
        }
    }
}

/// Release cached Metal buffers and other allocator caches.
///
/// Wraps `mlx::core::clear_cache()` (equivalent to `mx.metal.clear_cache()`
/// in Python). Call periodically during long generation loops to free
/// accumulated temporary Metal allocations.
pub fn clear_cache() {
    unsafe {
        mlx_sys::mlx_metal_clear_cache();
    }
    panic_if_mlx_error("mlx_metal_clear_cache");
}

pub fn active_memory_bytes() -> u64 {
    let value = unsafe { mlx_sys::mlx_get_active_memory() as u64 };
    panic_if_mlx_error("mlx_get_active_memory");
    value
}

pub fn peak_memory_bytes() -> u64 {
    let value = unsafe { mlx_sys::mlx_get_peak_memory() as u64 };
    panic_if_mlx_error("mlx_get_peak_memory");
    value
}

pub fn cache_memory_bytes() -> u64 {
    let value = unsafe { mlx_sys::mlx_get_cache_memory() as u64 };
    panic_if_mlx_error("mlx_get_cache_memory");
    value
}

pub fn set_memory_limit_bytes(limit: u64) -> u64 {
    let previous = unsafe { mlx_sys::mlx_set_memory_limit(limit as usize) as u64 };
    panic_if_mlx_error("mlx_set_memory_limit");
    previous
}

pub fn set_cache_limit_bytes(limit: u64) -> u64 {
    let previous = unsafe { mlx_sys::mlx_set_cache_limit(limit as usize) as u64 };
    panic_if_mlx_error("mlx_set_cache_limit");
    previous
}

pub fn set_wired_limit_bytes(limit: u64) -> u64 {
    let previous = unsafe { mlx_sys::mlx_set_wired_limit(limit as usize) as u64 };
    panic_if_mlx_error("mlx_set_wired_limit");
    previous
}
