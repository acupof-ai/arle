//! Direct FFI bindings to MLX C++ API via C bridge.
//!
//! No mlx-c intermediate layer. `mlx_array` is an opaque pointer to
//! `mlx::core::array*` (reinterpret_cast, no wrapper struct).
//!
//! All functions are `extern "C"` — defined in `src/mlx_bridge.cpp`.

#![allow(non_camel_case_types)]

use std::sync::{Mutex, MutexGuard};

static MLX_GUARD: Mutex<()> = Mutex::new(());

/// Process-wide guard for MLX FFI calls that mutate or evaluate MLX global
/// state. MLX's default device/stream and allocator are process-global, so
/// Rust callers that need serialization must share this guard across crates.
pub fn mlx_guard() -> MutexGuard<'static, ()> {
    MLX_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

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
pub const MLX_UINT64: i32 = 4;
pub const MLX_INT8: i32 = 5;
pub const MLX_INT16: i32 = 6;
pub const MLX_INT32: i32 = 7;
pub const MLX_INT64: i32 = 8;
pub const MLX_FLOAT16: i32 = 9;
pub const MLX_FLOAT32: i32 = 10;
pub const MLX_FLOAT64: i32 = 11;
pub const MLX_BFLOAT16: i32 = 12;
pub const MLX_COMPLEX64: i32 = 13;

unsafe extern "C" {
    // === Error handling ===

    /// Returns the last error message, or null if no error.
    /// Thread-local — safe to call from any thread.
    pub fn mlx_last_error() -> *const std::ffi::c_char;

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
    pub fn mlx_array_data_int32(a: *mut mlx_array) -> *const i32;
    pub fn mlx_array_size(a: *mut mlx_array) -> usize;

    // === Binary ops ===

    pub fn mlx_add(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_subtract(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_multiply(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_matmul(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_greater(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_prefix_match_len_i32(lhs: *mut mlx_array, rhs: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_prefix_match_len_i32_batched(
        lhs: *mut mlx_array,
        rhs: *mut mlx_array,
    ) -> *mut mlx_array;
    pub fn mlx_gather_axis1_i32(values: *mut mlx_array, indices: *mut mlx_array) -> *mut mlx_array;

    // === Unary ops ===

    pub fn mlx_exp(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_log1p(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_negative(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_sqrt(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_reciprocal(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_sigmoid(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_tanh(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_erf(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_log(a: *mut mlx_array) -> *mut mlx_array;

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

    /// Scatter-add into a zero-initialized `[vocab, feature_dim]` output.
    /// For each i in 0..prefix_rows, adds `updates_data[i*feature_dim..][..feature_dim]`
    /// into row `indices_data[i]`. Indices must already be in-bounds (the
    /// caller is responsible for OOB/negative filtering — the C++ helper
    /// does NOT sanitize).
    pub fn mlx_scatter_add_rows_f32(
        updates_data: *const f32,
        indices_data: *const i32,
        prefix_rows: i32,
        feature_dim: i32,
        vocab: i32,
    ) -> *mut mlx_array;

    // === Reduction ===

    pub fn mlx_sum_axis(a: *mut mlx_array, axis: i32, keepdims: bool) -> *mut mlx_array;
    pub fn mlx_mean_axis(a: *mut mlx_array, axis: i32, keepdims: bool) -> *mut mlx_array;
    pub fn mlx_max_axis(a: *mut mlx_array, axis: i32, keepdims: bool) -> *mut mlx_array;
    pub fn mlx_logsumexp_axis(a: *mut mlx_array, axis: i32, keepdims: bool) -> *mut mlx_array;
    pub fn mlx_softmax_axis(a: *mut mlx_array, axis: i32, precise: bool) -> *mut mlx_array;
    pub fn mlx_argmax(a: *mut mlx_array, keepdims: bool) -> *mut mlx_array;
    pub fn mlx_argmax_axis(a: *mut mlx_array, axis: i32, keepdims: bool) -> *mut mlx_array;

    // === Quantized ===

    pub fn mlx_fused_quantized_gated_mlp(
        x: *mut mlx_array,
        gate_w: *mut mlx_array,
        gate_s: *mut mlx_array,
        gate_b: *mut mlx_array,
        up_w: *mut mlx_array,
        up_s: *mut mlx_array,
        up_b: *mut mlx_array,
        down_w: *mut mlx_array,
        down_s: *mut mlx_array,
        down_b: *mut mlx_array,
        group_size: i32,
        bits: i32,
    ) -> *mut mlx_array;

    pub fn mlx_quantized_matmul(
        x: *mut mlx_array,
        w: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array,
        transpose: bool,
        group_size: i32,
        bits: i32,
    ) -> *mut mlx_array;
    pub fn mlx_quantize(
        w: *mut mlx_array,
        group_size: i32,
        bits: i32,
        out_w: *mut *mut mlx_array,
        out_scales: *mut *mut mlx_array,
        out_biases: *mut *mut mlx_array,
    ) -> i32;
    pub fn mlx_dequantize(
        w: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array,
        group_size: i32,
        bits: i32,
    ) -> *mut mlx_array;
    pub fn mlx_gguf_quantized_matmul(
        x: *mut mlx_array,
        w: *mut mlx_array,
        format: i32,
        rows: i32,
        cols: i32,
    ) -> *mut mlx_array;
    pub fn mlx_gguf_embedding(
        ids: *mut mlx_array,
        w: *mut mlx_array,
        format: i32,
        rows: i32,
        cols: i32,
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

    // === Compiled DFlash draft model ===

    pub fn dflash_draft_new() -> *mut std::ffi::c_void;
    pub fn dflash_draft_free(model: *mut std::ffi::c_void);
    pub fn dflash_draft_set_config(
        model: *mut std::ffi::c_void,
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        num_layers: i32,
        rope_theta: f32,
        rms_eps: f32,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_draft_push_layer(
        model: *mut std::ffi::c_void,
        q_w: *mut mlx_array,
        q_s: *mut mlx_array,
        q_b: *mut mlx_array,
        q_gs: i32,
        q_bits: i32,
        k_w: *mut mlx_array,
        k_s: *mut mlx_array,
        k_b: *mut mlx_array,
        k_gs: i32,
        k_bits: i32,
        v_w: *mut mlx_array,
        v_s: *mut mlx_array,
        v_b: *mut mlx_array,
        v_gs: i32,
        v_bits: i32,
        o_w: *mut mlx_array,
        o_s: *mut mlx_array,
        o_b: *mut mlx_array,
        o_gs: i32,
        o_bits: i32,
        gate_w: *mut mlx_array,
        gate_s: *mut mlx_array,
        gate_b: *mut mlx_array,
        gate_gs: i32,
        gate_bits: i32,
        up_w: *mut mlx_array,
        up_s: *mut mlx_array,
        up_b: *mut mlx_array,
        up_gs: i32,
        up_bits: i32,
        down_w: *mut mlx_array,
        down_s: *mut mlx_array,
        down_b: *mut mlx_array,
        down_gs: i32,
        down_bits: i32,
        input_norm: *mut mlx_array,
        post_attn_norm: *mut mlx_array,
        q_norm: *mut mlx_array,
        k_norm: *mut mlx_array,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_draft_set_fc_norms(
        model: *mut std::ffi::c_void,
        fc_w: *mut mlx_array,
        fc_s: *mut mlx_array,
        fc_b: *mut mlx_array,
        fc_gs: i32,
        fc_bits: i32,
        hidden_norm: *mut mlx_array,
        norm: *mut mlx_array,
    );
    pub fn dflash_draft_finalize(model: *mut std::ffi::c_void) -> i32;
    pub fn dflash_draft_forward(
        model: *mut std::ffi::c_void,
        noise_embedding: *mut mlx_array,
        target_hidden: *mut mlx_array,
        kv_caches: *mut *mut mlx_array,
        n_kv: i32,
        rope_offset: i32,
        out_hidden: *mut *mut mlx_array,
        out_kv_caches: *mut *mut mlx_array,
    ) -> i32;
    pub fn dflash_draft_forward_batched(
        model: *mut std::ffi::c_void,
        noise_embedding: *mut mlx_array,
        target_hidden: *mut mlx_array,
        batch_size: i32,
        q_offsets: *mut mlx_array,
        k_offsets: *mut mlx_array,
        kv_caches: *mut *mut mlx_array,
        n_kv: i32,
        attn_mask: *mut mlx_array,
        out_hidden: *mut *mut mlx_array,
        out_kv_caches: *mut *mut mlx_array,
    ) -> i32;

    // === Compiled Qwen3.5 model ===

    pub fn qwen35_compiled_new() -> *mut std::ffi::c_void;
    pub fn qwen35_compiled_free(model: *mut std::ffi::c_void);
    pub fn qwen35_compiled_set_gdr_metal_kernel_enabled(model: *mut std::ffi::c_void, enabled: i32);
    pub fn qwen35_compiled_add_dense_weight(model: *mut std::ffi::c_void, w: *mut mlx_array)
    -> i32;
    pub fn qwen35_compiled_add_affine_weight(
        model: *mut std::ffi::c_void,
        w: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array,
        group_size: i32,
        bits: i32,
    ) -> i32;
    pub fn qwen35_compiled_add_gguf_weight(
        model: *mut std::ffi::c_void,
        w: *mut mlx_array,
        format: i32,
        rows: i32,
        cols: i32,
    ) -> i32;
    pub fn qwen35_compiled_add_gguf_input_reordered_weight(
        model: *mut std::ffi::c_void,
        w: *mut mlx_array,
        format: i32,
        rows: i32,
        cols: i32,
        num_key_heads: i32,
        num_value_heads_per_key: i32,
        head_dim: i32,
    ) -> i32;
    pub fn qwen35_compiled_set_config(
        model: *mut std::ffi::c_void,
        rope_theta: f32,
        rms_eps: f32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        hidden_size: i32,
    );
    /// Declare whether Q has the gated half (Qwen3.5 = 1, Qwen3 = 0).
    /// Must be called before `qwen35_compiled_finalize`.
    pub fn qwen35_compiled_set_qk_gate(model: *mut std::ffi::c_void, enabled: i32);
    pub fn qwen35_compiled_set_embed(
        model: *mut std::ffi::c_void,
        embed_tokens: *mut mlx_array,
        final_norm_w: *mut mlx_array,
        lm_head_w: *mut mlx_array,
        lm_head_s: *mut mlx_array,
        lm_head_b: *mut mlx_array,
        lm_gs: i32,
        lm_bits: i32,
    );
    pub fn qwen35_compiled_set_embed_v2(
        model: *mut std::ffi::c_void,
        embed_tokens: *mut mlx_array,
        final_norm_w: *mut mlx_array,
        lm_head_id: i32,
    );
    pub fn qwen35_compiled_set_packed_embed_v2(
        model: *mut std::ffi::c_void,
        embed_id: i32,
        final_norm_w: *mut mlx_array,
        lm_head_id: i32,
    );
    /// Set quantized embed weights for as_linear lm_head (tie_word_embeddings).
    pub fn qwen35_compiled_set_embed_as_linear(
        model: *mut std::ffi::c_void,
        w: *mut mlx_array,
        s: *mut mlx_array,
        b: *mut mlx_array,
        gs: i32,
        bits: i32,
    );
    pub fn qwen35_compiled_set_embed_as_linear_v2(model: *mut std::ffi::c_void, embed_id: i32);
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_push_full_attn(
        model: *mut std::ffi::c_void,
        input_ln: *mut mlx_array,
        post_ln: *mut mlx_array,
        q_w: *mut mlx_array,
        q_s: *mut mlx_array,
        q_b: *mut mlx_array,
        q_gs: i32,
        q_bits: i32,
        k_w: *mut mlx_array,
        k_s: *mut mlx_array,
        k_b: *mut mlx_array,
        v_w: *mut mlx_array,
        v_s: *mut mlx_array,
        v_b: *mut mlx_array,
        o_w: *mut mlx_array,
        o_s: *mut mlx_array,
        o_b: *mut mlx_array,
        q_norm: *mut mlx_array,
        k_norm: *mut mlx_array,
        gu_w: *mut mlx_array,
        gu_s: *mut mlx_array,
        gu_b: *mut mlx_array,
        gu_gs: i32,
        gu_bits: i32,
        gate_dim: i32,
        dw_w: *mut mlx_array,
        dw_s: *mut mlx_array,
        dw_b: *mut mlx_array,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_push_full_attn_v2(
        model: *mut std::ffi::c_void,
        input_ln: *mut mlx_array,
        post_ln: *mut mlx_array,
        q_id: i32,
        k_id: i32,
        v_id: i32,
        o_id: i32,
        q_norm: *mut mlx_array,
        k_norm: *mut mlx_array,
        gate_up_id: i32,
        gate_dim: i32,
        down_id: i32,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_push_gdr(
        model: *mut std::ffi::c_void,
        input_ln: *mut mlx_array,
        post_ln: *mut mlx_array,
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
        conv_kernel: i32,
        a_log: *mut mlx_array,
        dt_bias: *mut mlx_array,
        norm_w: *mut mlx_array,
        gdr_rms_eps: f32,
        out_w: *mut mlx_array,
        out_s: *mut mlx_array,
        out_b: *mut mlx_array,
        out_gs: i32,
        out_bits: i32,
        num_key_heads: i32,
        key_dim: i32,
        num_value_heads: i32,
        value_dim: i32,
        gu_w: *mut mlx_array,
        gu_s: *mut mlx_array,
        gu_b: *mut mlx_array,
        gu_gs: i32,
        gu_bits: i32,
        gate_dim: i32,
        dw_w: *mut mlx_array,
        dw_s: *mut mlx_array,
        dw_b: *mut mlx_array,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_push_gdr_v2(
        model: *mut std::ffi::c_void,
        input_ln: *mut mlx_array,
        post_ln: *mut mlx_array,
        qkvz_id: i32,
        qkv_split: i32,
        z_split: i32,
        ba_id: i32,
        ba_num_heads: i32,
        conv1d_w: *mut mlx_array,
        conv_kernel: i32,
        a_log: *mut mlx_array,
        dt_bias: *mut mlx_array,
        norm_w: *mut mlx_array,
        gdr_rms_eps: f32,
        out_id: i32,
        num_key_heads: i32,
        key_dim: i32,
        num_value_heads: i32,
        value_dim: i32,
        gate_up_id: i32,
        gate_dim: i32,
        down_id: i32,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_set_last_moe_mlp(
        model: *mut std::ffi::c_void,
        router_w: *mut mlx_array,
        router_s: *mut mlx_array,
        router_b: *mut mlx_array,
        router_gs: i32,
        router_bits: i32,
        expert_gate_w: *mut mlx_array,
        expert_gate_s: *mut mlx_array,
        expert_gate_b: *mut mlx_array,
        expert_up_w: *mut mlx_array,
        expert_up_s: *mut mlx_array,
        expert_up_b: *mut mlx_array,
        expert_down_w: *mut mlx_array,
        expert_down_s: *mut mlx_array,
        expert_down_b: *mut mlx_array,
        expert_gs: i32,
        expert_bits: i32,
        shared_gate_w: *mut mlx_array,
        shared_gate_s: *mut mlx_array,
        shared_gate_b: *mut mlx_array,
        shared_up_w: *mut mlx_array,
        shared_up_s: *mut mlx_array,
        shared_up_b: *mut mlx_array,
        shared_down_w: *mut mlx_array,
        shared_down_s: *mut mlx_array,
        shared_down_b: *mut mlx_array,
        shared_gate_router_w: *mut mlx_array,
        shared_gate_router_s: *mut mlx_array,
        shared_gate_router_b: *mut mlx_array,
        num_experts: i32,
        top_k: i32,
        norm_topk_prob: bool,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_set_separate_proj(
        model: *mut std::ffi::c_void,
        qkv_w: *mut mlx_array,
        qkv_s: *mut mlx_array,
        qkv_b: *mut mlx_array,
        qkv_gs: i32,
        qkv_bits: i32,
        z_w: *mut mlx_array,
        z_s: *mut mlx_array,
        z_b: *mut mlx_array,
        b_w: *mut mlx_array,
        b_s: *mut mlx_array,
        b_b: *mut mlx_array,
        a_w: *mut mlx_array,
        a_s: *mut mlx_array,
        a_b: *mut mlx_array,
        gate_w: *mut mlx_array,
        gate_s: *mut mlx_array,
        gate_b: *mut mlx_array,
        mlp_gs: i32,
        mlp_bits: i32,
        up_w: *mut mlx_array,
        up_s: *mut mlx_array,
        up_b: *mut mlx_array,
    );
    pub fn qwen35_compiled_set_separate_proj_v2(
        model: *mut std::ffi::c_void,
        qkv_id: i32,
        z_id: i32,
        b_id: i32,
        a_id: i32,
        gate_id: i32,
        up_id: i32,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_set_separate_mlp(
        model: *mut std::ffi::c_void,
        gate_w: *mut mlx_array,
        gate_s: *mut mlx_array,
        gate_b: *mut mlx_array,
        mlp_gs: i32,
        mlp_bits: i32,
        up_w: *mut mlx_array,
        up_s: *mut mlx_array,
        up_b: *mut mlx_array,
    );
    pub fn qwen35_compiled_set_separate_mlp_v2(
        model: *mut std::ffi::c_void,
        gate_id: i32,
        up_id: i32,
    );
    pub fn qwen35_compiled_finalize(model: *mut std::ffi::c_void) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_step(
        model: *mut std::ffi::c_void,
        token_id: *mut mlx_array,
        cache_pos: i32,
        kv_caches: *mut *mut mlx_array,
        n_kv: i32,
        gdr_states: *mut *mut mlx_array,
        n_gdr: i32,
        out_logits: *mut *mut mlx_array,
        out_kv_caches: *mut *mut mlx_array,
        out_gdr_states: *mut *mut mlx_array,
    ) -> i32;
    pub fn qwen35_session_begin(
        model: *mut std::ffi::c_void,
        kv_caches: *mut *mut mlx_array,
        n_kv: i32,
        gdr_states: *mut *mut mlx_array,
        n_gdr: i32,
    ) -> i32;
    pub fn qwen35_session_end(
        model: *mut std::ffi::c_void,
        out_kv_caches: *mut *mut mlx_array,
        n_kv: i32,
        out_gdr_states: *mut *mut mlx_array,
        n_gdr: i32,
    ) -> i32;
    pub fn qwen35_compiled_step_session(
        model: *mut std::ffi::c_void,
        token_id: *mut mlx_array,
        cache_pos: i32,
        out_logits: *mut *mut mlx_array,
    ) -> i32;
    pub fn qwen35_compiled_prefill_session(
        model: *mut std::ffi::c_void,
        token_ids: *mut mlx_array,
        prompt_len: i32,
        cache_pos: i32,
        out_logits: *mut *mut mlx_array,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_step_batch(
        model: *mut std::ffi::c_void,
        token_ids: *mut mlx_array,
        batch_size: i32,
        cache_pos: i32,
        kv_caches: *mut *mut mlx_array,
        n_kv_per_request: i32,
        gdr_states: *mut *mut mlx_array,
        n_gdr_per_request: i32,
        attn_mask: *mut mlx_array,
        rope_offsets: *mut mlx_array,
        out_logits: *mut *mut mlx_array,
        out_kv_caches: *mut *mut mlx_array,
        out_gdr_states: *mut *mut mlx_array,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_step_batch_packed(
        model: *mut std::ffi::c_void,
        token_ids: *mut mlx_array,
        batch_size: i32,
        cache_pos: i32,
        packed_kv_caches: *mut *mut mlx_array,
        n_kv: i32,
        packed_gdr_states: *mut *mut mlx_array,
        n_gdr: i32,
        attn_mask: *mut mlx_array,
        rope_offsets: *mut mlx_array,
        out_logits: *mut *mut mlx_array,
        out_packed_kv_caches: *mut *mut mlx_array,
        out_packed_gdr_states: *mut *mut mlx_array,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_prefill(
        model: *mut std::ffi::c_void,
        token_ids: *mut mlx_array,
        prompt_len: i32,
        cache_pos: i32,
        kv_caches: *mut *mut mlx_array,
        n_kv: i32,
        gdr_states: *mut *mut mlx_array,
        n_gdr: i32,
        out_logits: *mut *mut mlx_array,
        out_kv_caches: *mut *mut mlx_array,
        out_gdr_states: *mut *mut mlx_array,
    ) -> i32;
    /// DFlash verify: parallel forward over a draft block, returning all-position
    /// logits [1, block_size, vocab]. Respects model-level tape_mode and capture
    /// layers — one call emits per-step GDR tapes and captured hidden for the
    /// entire block, replacing the previous 16 × seq_len=1 sequential verify loop.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_verify_block(
        model: *mut std::ffi::c_void,
        token_ids: *mut mlx_array,
        block_size: i32,
        cache_pos: i32,
        kv_caches: *mut *mut mlx_array,
        n_kv: i32,
        gdr_states: *mut *mut mlx_array,
        n_gdr: i32,
        out_logits: *mut *mut mlx_array,
        out_kv_caches: *mut *mut mlx_array,
        out_gdr_states: *mut *mut mlx_array,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_verify_block_summary(
        model: *mut std::ffi::c_void,
        token_ids: *mut mlx_array,
        block_size: i32,
        cache_pos: i32,
        kv_caches: *mut *mut mlx_array,
        n_kv: i32,
        gdr_states: *mut *mut mlx_array,
        n_gdr: i32,
        temperature: f32,
        greedy: bool,
        suppress_token_id: i32,
        out_matched_prefix_len: *mut i32,
        out_next_token: *mut i32,
        out_kv_caches: *mut *mut mlx_array,
        out_gdr_states: *mut *mut mlx_array,
    ) -> i32;
    pub fn qwen35_compiled_verify_block_batched(
        model: *mut std::ffi::c_void,
        token_ids: *mut mlx_array,
        batch_size: i32,
        block_size: i32,
        cache_pos_arr: *const i32,
        packed_kv_caches: *mut *mut mlx_array,
        n_kv: i32,
        packed_gdr_states: *mut *mut mlx_array,
        n_gdr: i32,
        attn_mask: *mut mlx_array,
        rope_offsets: *mut mlx_array,
        out_logits: *mut *mut mlx_array,
        out_packed_kv_caches: *mut *mut mlx_array,
        out_packed_gdr_states: *mut *mut mlx_array,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_verify_block_batched_sampled(
        model: *mut std::ffi::c_void,
        token_ids: *mut mlx_array,
        batch_size: i32,
        block_size: i32,
        cache_pos_arr: *const i32,
        packed_kv_caches: *mut *mut mlx_array,
        n_kv: i32,
        packed_gdr_states: *mut *mut mlx_array,
        n_gdr: i32,
        attn_mask: *mut mlx_array,
        rope_offsets: *mut mlx_array,
        temperature: f32,
        greedy: bool,
        suppress_token_id: i32,
        out_sampled: *mut *mut mlx_array,
        out_packed_kv_caches: *mut *mut mlx_array,
        out_packed_gdr_states: *mut *mut mlx_array,
    ) -> i32;

    /// Full decode loop in C++ — all intermediates stay alive within the loop.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_generate(
        model: *mut std::ffi::c_void,
        prompt_ids: *const i32,
        prompt_len: i32,
        max_new_tokens: i32,
        temperature: f32,
        greedy: bool,
        out_tokens: *mut i32,
        out_count: *mut i32,
        out_prefill_ms: *mut f64,
        out_decode_ms: *mut f64,
        on_token: Option<unsafe extern "C" fn(i32, *mut std::ffi::c_void) -> i32>,
        callback_ctx: *mut std::ffi::c_void,
        stop_tokens: *const i32,
        n_stop_tokens: i32,
    ) -> i32;

    // === Qwen3.6 MoE block ===

    /// Qwen3.5/3.6 SparseMoeBlock forward (Metal only).
    ///
    /// Composes MLX ops to reproduce `Qwen3NextSparseMoeBlock.__call__` in one
    /// C++ call: 8-bit-quantized router → top-k (argpartition + slice) →
    /// take_along_axis scores → optional norm_topk_prob → SwitchGLU over the
    /// switch-mlp experts (4-bit quantized, stacked) → weighted sum over top_k
    /// → dense shared expert (4-bit quantized SwiGLU) gated by an 8-bit scalar
    /// router → sum.
    ///
    /// All `*_w/*_scales/*_biases` triples are mlx quantized-linear triples in
    /// affine mode (`group_size` = 64 for Qwen3.6-A3B, `bits` = 4 for experts
    /// and 8 for both routers per mlx-community config).
    ///
    /// Expert weights are stacked on the expert axis:
    /// `expert_{gate,up}_w : [E, Hmoe, H/pack]`,
    /// `expert_down_w : [E, H, Hmoe/pack]`. Shared-expert weights are plain
    /// 2-D quantized linears matching `mlx_fused_quantized_gated_mlp`.
    ///
    /// Returns a newly-allocated array handle (caller must `mlx_array_free`)
    /// or nullptr on failure (check `mlx_last_error()`).
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_moe_block_forward(
        hidden: *mut mlx_array,
        router_w: *mut mlx_array,
        router_scales: *mut mlx_array,
        router_biases: *mut mlx_array,
        router_bits: i32,
        router_group_size: i32,
        expert_gate_w: *mut mlx_array,
        expert_gate_scales: *mut mlx_array,
        expert_gate_biases: *mut mlx_array,
        expert_up_w: *mut mlx_array,
        expert_up_scales: *mut mlx_array,
        expert_up_biases: *mut mlx_array,
        expert_down_w: *mut mlx_array,
        expert_down_scales: *mut mlx_array,
        expert_down_biases: *mut mlx_array,
        expert_bits: i32,
        expert_group_size: i32,
        shared_gate_w: *mut mlx_array,
        shared_gate_scales: *mut mlx_array,
        shared_gate_biases: *mut mlx_array,
        shared_up_w: *mut mlx_array,
        shared_up_scales: *mut mlx_array,
        shared_up_biases: *mut mlx_array,
        shared_down_w: *mut mlx_array,
        shared_down_scales: *mut mlx_array,
        shared_down_biases: *mut mlx_array,
        shared_gate_router_w: *mut mlx_array,
        shared_gate_router_scales: *mut mlx_array,
        shared_gate_router_biases: *mut mlx_array,
        num_experts: i32,
        top_k: i32,
        norm_topk_prob: bool,
    ) -> *mut mlx_array;

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
    pub fn mlx_fast_rope_dynamic(
        x: *mut mlx_array,
        dims: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: *mut mlx_array,
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
    pub fn mlx_fast_sdpa_masked(
        q: *mut mlx_array,
        k: *mut mlx_array,
        v: *mut mlx_array,
        scale: f32,
        mask: *mut mlx_array,
    ) -> *mut mlx_array;
    #[allow(clippy::too_many_arguments)]
    pub fn mlx_gated_delta_with_tape(
        q: *mut mlx_array,
        k: *mut mlx_array,
        v: *mut mlx_array,
        g: *mut mlx_array,
        beta: *mut mlx_array,
        state_in: *mut mlx_array,
        t: i32,
        out_y: *mut *mut mlx_array,
        out_state: *mut *mut mlx_array,
        out_tape: *mut *mut mlx_array,
    );
    pub fn mlx_tape_replay(
        tape: *mut mlx_array,
        k: *mut mlx_array,
        g: *mut mlx_array,
        state_in: *mut mlx_array,
        steps: i32,
    ) -> *mut mlx_array;
    pub fn mlx_tape_replay_varlen(
        tape: *mut mlx_array,
        k: *mut mlx_array,
        g: *mut mlx_array,
        state_in: *mut mlx_array,
        steps: *mut mlx_array,
    ) -> *mut mlx_array;
    pub fn mlx_batched_sdpa_2pass(
        queries: *mut mlx_array,
        keys: *mut mlx_array,
        values: *mut mlx_array,
        scale: f32,
        gqa_factor: i32,
    ) -> *mut mlx_array;

    // === Qwen3.5 DFlash support ===

    pub fn qwen35_set_tape_mode(model: *mut std::ffi::c_void, enabled: bool);
    pub fn qwen35_get_tape_count(model: *mut std::ffi::c_void) -> i32;
    pub fn qwen35_get_tape(
        model: *mut std::ffi::c_void,
        idx: i32,
        out_tape: *mut *mut mlx_array,
        out_k: *mut *mut mlx_array,
        out_g: *mut *mut mlx_array,
        out_qkv: *mut *mut mlx_array,
    ) -> i32;
    pub fn qwen35_read_and_clear_gdr_tapes(
        model: *mut std::ffi::c_void,
        out_tapes: *mut *mut mlx_array,
        out_k: *mut *mut mlx_array,
        out_g: *mut *mut mlx_array,
        out_qkv: *mut *mut mlx_array,
        capacity: i32,
    ) -> i32;
    pub fn qwen35_set_capture_layers(
        model: *mut std::ffi::c_void,
        layer_ids: *const i32,
        count: i32,
    );
    pub fn qwen35_get_captured_hidden_count(model: *mut std::ffi::c_void) -> i32;
    pub fn qwen35_get_captured_hidden(
        model: *mut std::ffi::c_void,
        idx: i32,
        out: *mut *mut mlx_array,
    ) -> i32;

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

    // === Memory management ===

    /// Current active MLX allocator memory in bytes.
    pub fn mlx_get_active_memory() -> usize;
    /// Peak MLX allocator memory in bytes.
    pub fn mlx_get_peak_memory() -> usize;
    /// Cached MLX allocator memory in bytes.
    pub fn mlx_get_cache_memory() -> usize;
    /// Set the MLX allocator memory limit in bytes. Returns the previous limit.
    pub fn mlx_set_memory_limit(limit: usize) -> usize;
    /// Set the MLX allocator cache limit in bytes. Returns the previous limit.
    pub fn mlx_set_cache_limit(limit: usize) -> usize;
    /// Set the MLX allocator wired limit in bytes. Returns the previous limit.
    pub fn mlx_set_wired_limit(limit: usize) -> usize;
    /// Release cached Metal buffers and other allocator caches.
    /// Equivalent to `mx.metal.clear_cache()` in Python.
    pub fn mlx_metal_clear_cache();
}
