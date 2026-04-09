//! Direct FFI bindings to MLX C++ API via C bridge.
//!
//! No mlx-c intermediate layer. `mlx_array` is an opaque pointer to
//! `mlx::core::array*` (reinterpret_cast, no wrapper struct).
//!
//! All functions are `extern "C"` — defined in `src/mlx_bridge.cpp`.

#![allow(non_camel_case_types)]

/// Opaque handle to `mlx::core::array`. All access through pointers.
#[repr(C)]
pub struct mlx_array {
    _opaque: [u8; 0],
}

// Dtype constants — must match mlx::core::Dtype::Val and mlx_common.h
pub const MLX_BOOL: i32 = 0;
pub const MLX_UINT8: i32 = 1;
pub const MLX_UINT16: i32 = 2;
pub const MLX_UINT32: i32 = 3;
pub const MLX_INT8: i32 = 5;
pub const MLX_INT16: i32 = 6;
pub const MLX_INT32: i32 = 7;
pub const MLX_INT64: i32 = 8;
pub const MLX_FLOAT16: i32 = 9;
pub const MLX_FLOAT32: i32 = 10;
pub const MLX_BFLOAT16: i32 = 12;
pub const MLX_COMPLEX64: i32 = 13;

unsafe extern "C" {
    // === Array lifecycle ===

    pub fn mlx_array_new_float32(val: f32) -> *mut mlx_array;
    pub fn mlx_array_new_int32(val: i32) -> *mut mlx_array;
    pub fn mlx_array_from_data(
        data: *const std::ffi::c_void,
        shape: *const i32,
        ndim: i32,
        dtype: i32,
    ) -> *mut mlx_array;
    /// Copy shared_ptr (increment refcount, same underlying data).
    pub fn mlx_array_clone(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_array_free(a: *mut mlx_array);
    pub fn mlx_array_ndim(a: *mut mlx_array) -> i32;
    /// Returns pointer to shape data. Valid while array is alive.
    pub fn mlx_array_shape(a: *mut mlx_array) -> *const i32;
    /// Returns dtype as integer (see MLX_* constants).
    pub fn mlx_array_dtype(a: *mut mlx_array) -> i32;
    /// Extract scalar i32 value (blocks until computed).
    pub fn mlx_array_item_int32(a: *mut mlx_array) -> i32;
    /// Extract scalar f32 value (blocks until computed).
    pub fn mlx_array_item_float32(a: *mut mlx_array) -> f32;
    /// Access the underlying data pointer (after eval). Caller must not free.
    pub fn mlx_array_data_float32(a: *mut mlx_array) -> *const f32;
    pub fn mlx_array_size(a: *mut mlx_array) -> usize;

    // === Binary ops ===

    pub fn mlx_add(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_subtract(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_multiply(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_matmul(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_greater(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;

    // === Unary ops ===

    pub fn mlx_exp(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_log1p(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_negative(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_sqrt(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_reciprocal(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_sigmoid(a: *mut mlx_array) -> *mut mlx_array;

    // === Shape ops ===

    pub fn mlx_reshape(a: *mut mlx_array, shape: *const i32, ndim: usize) -> *mut mlx_array;
    /// Reverse all axes.
    pub fn mlx_transpose(a: *mut mlx_array) -> *mut mlx_array;
    /// Transpose with explicit axis permutation.
    pub fn mlx_transpose_axes(a: *mut mlx_array, axes: *const i32, n: usize) -> *mut mlx_array;
    pub fn mlx_astype(a: *mut mlx_array, dtype: i32) -> *mut mlx_array;
    pub fn mlx_broadcast_to(a: *mut mlx_array, shape: *const i32, ndim: usize) -> *mut mlx_array;
    pub fn mlx_expand_dims(a: *mut mlx_array, axis: i32) -> *mut mlx_array;
    pub fn mlx_zeros(shape: *const i32, ndim: usize, dtype: i32) -> *mut mlx_array;

    // === Indexing ===

    pub fn mlx_take_axis(a: *mut mlx_array, indices: *mut mlx_array, axis: i32) -> *mut mlx_array;
    pub fn mlx_slice(
        a: *mut mlx_array,
        start: *const i32,
        stop: *const i32,
        strides: *const i32,
        ndim: usize,
    ) -> *mut mlx_array;
    pub fn mlx_slice_update(
        src: *mut mlx_array,
        update: *mut mlx_array,
        start: *const i32,
        stop: *const i32,
        strides: *const i32,
        ndim: usize,
    ) -> *mut mlx_array;
    pub fn mlx_concatenate_axis(
        arrays: *mut *mut mlx_array,
        count: usize,
        axis: i32,
    ) -> *mut mlx_array;
    pub fn mlx_where(cond: *mut mlx_array, a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;

    // === Reduction ===

    pub fn mlx_sum_axis(a: *mut mlx_array, axis: i32, keepdims: bool) -> *mut mlx_array;
    pub fn mlx_argmax(a: *mut mlx_array, keepdims: bool) -> *mut mlx_array;

    // === Quantized ===

    pub fn mlx_quantized_matmul(
        x: *mut mlx_array,
        w: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array,
        transpose: bool,
        group_size: i32,
        bits: i32,
    ) -> *mut mlx_array;
    pub fn mlx_dequantize(
        w: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array,
        group_size: i32,
        bits: i32,
    ) -> *mut mlx_array;

    // === Contiguous ===

    pub fn mlx_contiguous(a: *mut mlx_array) -> *mut mlx_array;

    // === Conv ===

    pub fn mlx_conv1d(
        input: *mut mlx_array,
        weight: *mut mlx_array,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> *mut mlx_array;

    // === Fused ops ===

    /// g = exp(-exp(A_log) * softplus(alpha + dt_bias))
    /// Fuses 10 ops into 1 C++ call.
    pub fn mlx_compute_g(
        a_log: *mut mlx_array,
        alpha: *mut mlx_array,
        dt_bias: *mut mlx_array,
    ) -> *mut mlx_array;

    /// Full GDR layer forward in C++ — eliminates ~40 FFI calls per layer.
    #[allow(clippy::too_many_arguments)]
    pub fn mlx_gdr_layer_forward(
        x: *mut mlx_array,
        qkvz_w: *mut mlx_array,
        qkvz_s: *mut mlx_array,
        qkvz_b: *mut mlx_array,
        qkvz_gs: i32,
        qkvz_bits: i32,
        qkv_split: i32,
        z_split: i32,
        ba_w: *mut mlx_array,
        ba_s: *mut mlx_array,
        ba_b: *mut mlx_array,
        ba_gs: i32,
        ba_bits: i32,
        ba_num_heads: i32,
        conv1d_w: *mut mlx_array,
        conv_state: *mut *mut mlx_array,
        conv_kernel: i32,
        a_log: *mut mlx_array,
        dt_bias: *mut mlx_array,
        norm_w: *mut mlx_array,
        rms_eps: f32,
        out_w: *mut mlx_array,
        out_s: *mut mlx_array,
        out_b: *mut mlx_array,
        out_gs: i32,
        out_bits: i32,
        num_key_heads: i32,
        key_dim: i32,
        num_value_heads: i32,
        value_dim: i32,
        q_scale: f32,
        k_scale: f32,
        gdr_state: *mut *mut mlx_array,
        metal_kernel: *mut std::ffi::c_void,
        use_metal_kernel: i32,
        out_result: *mut *mut mlx_array,
    );

    // === Fast ops ===

    /// RMS normalization. Pass null for weight to use no learnable weight.
    pub fn mlx_fast_rms_norm(x: *mut mlx_array, weight: *mut mlx_array, eps: f32)
    -> *mut mlx_array;
    pub fn mlx_fast_rope(
        x: *mut mlx_array,
        dims: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: i32,
    ) -> *mut mlx_array;
    /// Scaled dot-product attention.
    /// mask_mode: "" for no mask, "causal" for causal masking.
    /// NEVER pass null — std::string(nullptr) is UB.
    pub fn mlx_fast_sdpa(
        q: *mut mlx_array,
        k: *mut mlx_array,
        v: *mut mlx_array,
        scale: f32,
        mask_mode: *const std::ffi::c_char,
    ) -> *mut mlx_array;

    // === Random ===

    pub fn mlx_random_categorical(logits: *mut mlx_array, axis: i32) -> *mut mlx_array;

    // === Transforms ===

    pub fn mlx_eval(arrays: *mut *mut mlx_array, count: usize);
    pub fn mlx_async_eval(arrays: *mut *mut mlx_array, count: usize);

    // === IO ===

    /// Load safetensors file. Returns count of loaded tensors.
    /// Names and arrays are written to out_names/out_arrays (caller must free via
    /// `mlx_free_loaded_tensors`).
    pub fn mlx_load_safetensors(
        path: *const std::ffi::c_char,
        out_names: *mut *mut *const std::ffi::c_char,
        out_arrays: *mut *mut *mut mlx_array,
    ) -> i32;
    pub fn mlx_free_loaded_tensors(
        names: *mut *const std::ffi::c_char,
        arrays: *mut *mut mlx_array,
        count: i32,
    );

    // === Metal kernel ===

    pub fn mlx_metal_kernel_new(
        name: *const std::ffi::c_char,
        input_names: *const *const std::ffi::c_char,
        n_inputs: usize,
        output_names: *const *const std::ffi::c_char,
        n_outputs: usize,
        source: *const std::ffi::c_char,
        header: *const std::ffi::c_char,
    ) -> *mut std::ffi::c_void;
    pub fn mlx_metal_kernel_free(kernel: *mut std::ffi::c_void);
    pub fn mlx_metal_kernel_apply(
        kernel: *mut std::ffi::c_void,
        inputs: *mut *mut mlx_array,
        n_inputs: usize,
        outputs: *mut *mut mlx_array,
        n_outputs: usize,
        grid: *const i32,
        threadgroup: *const i32,
        output_shapes: *const i32,
        output_shape_dims: *const i32,
        output_dtypes: *const i32,
        template_names: *const *const std::ffi::c_char,
        template_int_vals: *const i32,
        template_dtype_vals: *const i32,
        n_int_templates: usize,
        n_dtype_templates: usize,
    );
}
