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

    /// Returns the last error message, or null if no error.
    /// Thread-local — safe to call from any thread.
    pub fn mlx_last_error() -> *const std::ffi::c_char;
    /// Returns Metal's recommended max GPU working set in bytes, or 0 when no
    /// system Metal device is available.
    pub fn mlx_metal_recommended_max_working_set_size() -> u64;

    pub fn mlx_array_new_float32(val: f32) -> *mut mlx_array;
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
    /// Access the underlying data pointer (after eval). Caller must not free.
    pub fn mlx_array_data_float32(a: *mut mlx_array) -> *const f32;
    pub fn mlx_array_data_int32(a: *mut mlx_array) -> *const i32;
    pub fn mlx_array_size(a: *mut mlx_array) -> usize;
    pub fn mlx_array_nbytes(a: *mut mlx_array) -> usize;
    pub fn mlx_array_export_bytes(
        a: *mut mlx_array,
        out: *mut std::ffi::c_void,
        out_len: usize,
    ) -> usize;

    pub fn mlx_add(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_subtract(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_multiply(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_matmul(a: *mut mlx_array, b: *mut mlx_array) -> *mut mlx_array;

    pub fn mlx_exp(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_negative(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_sqrt(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_reciprocal(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_sigmoid(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_tanh(a: *mut mlx_array) -> *mut mlx_array;
    pub fn mlx_erf(a: *mut mlx_array) -> *mut mlx_array;

    pub fn mlx_reshape(a: *mut mlx_array, shape: *const i32, ndim: usize) -> *mut mlx_array;
    /// Reverse all axes.
    pub fn mlx_transpose(a: *mut mlx_array) -> *mut mlx_array;
    /// Transpose with explicit axis permutation.
    pub fn mlx_transpose_axes(a: *mut mlx_array, axes: *const i32, n: usize) -> *mut mlx_array;
    pub fn mlx_astype(a: *mut mlx_array, dtype: i32) -> *mut mlx_array;
    pub fn mlx_broadcast_to(a: *mut mlx_array, shape: *const i32, ndim: usize) -> *mut mlx_array;
    pub fn mlx_zeros(shape: *const i32, ndim: usize, dtype: i32) -> *mut mlx_array;

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

    pub fn mlx_sum_axis(a: *mut mlx_array, axis: i32, keepdims: bool) -> *mut mlx_array;
    pub fn mlx_mean_axis(a: *mut mlx_array, axis: i32, keepdims: bool) -> *mut mlx_array;
    pub fn mlx_logsumexp_axis(a: *mut mlx_array, axis: i32, keepdims: bool) -> *mut mlx_array;
    pub fn mlx_softmax_axis(a: *mut mlx_array, axis: i32, precise: bool) -> *mut mlx_array;
    pub fn mlx_argmax(a: *mut mlx_array, keepdims: bool) -> *mut mlx_array;
    pub fn mlx_argmax_axis(a: *mut mlx_array, axis: i32, keepdims: bool) -> *mut mlx_array;

    pub fn mlx_quantized_matmul(
        x: *mut mlx_array,
        w: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array,
        transpose: bool,
        group_size: i32,
        bits: i32,
        mode: i32,
    ) -> *mut mlx_array;
    pub fn mlx_dequantize(
        w: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array,
        group_size: i32,
        bits: i32,
        mode: i32,
    ) -> *mut mlx_array;

    pub fn mlx_contiguous(a: *mut mlx_array) -> *mut mlx_array;

    /// Full GDR layer forward in C++ — eliminates ~40 FFI calls per layer.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_draft_new() -> *mut std::ffi::c_void;
    pub fn dflash_draft_free(model: *mut std::ffi::c_void);
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_draft_set_config(
        model: *mut std::ffi::c_void,
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        num_layers: i32,
        rotary_dim: i32,
        attn_output_gate: i32,
        draft_kind: i32,
        rope_theta: f32,
        rms_eps: f32,
    );
    pub fn dflash_draft_set_qwen35_mtp_norms(
        model: *mut std::ffi::c_void,
        pre_fc_norm_embedding: *mut mlx_array,
        pre_fc_norm_hidden: *mut mlx_array,
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

    pub fn diffusion_gemma_new() -> *mut std::ffi::c_void;
    pub fn diffusion_gemma_free(model: *mut std::ffi::c_void);
    pub fn diffusion_gemma_add_dense_weight(model: *mut std::ffi::c_void, w: *mut mlx_array)
    -> i32;
    pub fn diffusion_gemma_add_affine_weight(
        model: *mut std::ffi::c_void,
        w: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array,
        group_size: i32,
        bits: i32,
    ) -> i32;
    pub fn diffusion_gemma_set_config(
        model: *mut std::ffi::c_void,
        hidden_size: i32,
        vocab_size: i32,
        rms_eps: f32,
        final_logit_softcap: f32,
    );
    pub fn diffusion_gemma_set_embed(
        model: *mut std::ffi::c_void,
        embed_id: i32,
        lm_head_id: i32,
        final_norm_id: i32,
    );
    pub fn diffusion_gemma_set_requires_self_conditioning(
        model: *mut std::ffi::c_void,
        required: bool,
    ) -> i32;
    pub fn diffusion_gemma_set_per_layer_embeddings(
        model: *mut std::ffi::c_void,
        embed_id: i32,
        projection_id: i32,
        norm_id: i32,
        num_layers: i32,
        hidden_size_per_layer_input: i32,
        vocab_size_per_layer_input: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn diffusion_gemma_set_vision_config(
        model: *mut std::ffi::c_void,
        image_token_id: i32,
        hidden_size: i32,
        intermediate_size: i32,
        num_layers: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        patch_size: i32,
        pooling_kernel_size: i32,
        default_output_length: i32,
        position_embedding_size: i32,
        rope_theta: f32,
        rms_eps: f32,
        use_clipping: bool,
        patch_proj_id: i32,
        position_embedding_id: i32,
        vision_projection_id: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn diffusion_gemma_push_vision_layer(
        model: *mut std::ffi::c_void,
        input_ln_id: i32,
        q_id: i32,
        q_input_min_id: i32,
        q_input_max_id: i32,
        q_output_min_id: i32,
        q_output_max_id: i32,
        k_id: i32,
        k_input_min_id: i32,
        k_input_max_id: i32,
        k_output_min_id: i32,
        k_output_max_id: i32,
        v_id: i32,
        v_input_min_id: i32,
        v_input_max_id: i32,
        v_output_min_id: i32,
        v_output_max_id: i32,
        o_id: i32,
        o_input_min_id: i32,
        o_input_max_id: i32,
        o_output_min_id: i32,
        o_output_max_id: i32,
        q_norm_id: i32,
        k_norm_id: i32,
        post_attn_ln_id: i32,
        pre_ff_ln_id: i32,
        gate_id: i32,
        gate_input_min_id: i32,
        gate_input_max_id: i32,
        gate_output_min_id: i32,
        gate_output_max_id: i32,
        up_id: i32,
        up_input_min_id: i32,
        up_input_max_id: i32,
        up_output_min_id: i32,
        up_output_max_id: i32,
        down_id: i32,
        down_input_min_id: i32,
        down_input_max_id: i32,
        down_output_min_id: i32,
        down_output_max_id: i32,
        post_ff_ln_id: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn diffusion_gemma_push_layer(
        model: *mut std::ffi::c_void,
        is_full_attention: bool,
        kv_shared_layer_index: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        rope_theta: f32,
        sliding_window: i32,
        input_ln_id: i32,
        q_id: i32,
        k_id: i32,
        v_id: i32,
        o_id: i32,
        q_norm_id: i32,
        k_norm_id: i32,
        post_attn_ln_id: i32,
        pre_ff_ln_id: i32,
        gate_id: i32,
        up_id: i32,
        down_id: i32,
        post_ff_ln_id: i32,
        pre_ff2_ln_id: i32,
        post_ff1_ln_id: i32,
        post_ff2_ln_id: i32,
        router_id: i32,
        router_scale_id: i32,
        per_expert_scale_id: i32,
        expert_gate_up_id: i32,
        expert_down_id: i32,
        layer_scalar_id: i32,
        num_experts: i32,
        top_k: i32,
    ) -> i32;
    pub fn diffusion_gemma_set_layer_ple(
        model: *mut std::ffi::c_void,
        layer_index: i32,
        gate_id: i32,
        projection_id: i32,
        norm_id: i32,
    ) -> i32;
    pub fn diffusion_gemma_set_self_conditioning(
        model: *mut std::ffi::c_void,
        pre_norm_id: i32,
        gate_id: i32,
        up_id: i32,
        down_id: i32,
    ) -> i32;
    pub fn diffusion_gemma_finalize(model: *mut std::ffi::c_void) -> i32;
    pub fn diffusion_gemma_begin_request(model: *mut std::ffi::c_void, seed: u64) -> i32;
    pub fn diffusion_gemma_prefill(
        model: *mut std::ffi::c_void,
        tokens: *const i32,
        len: i32,
    ) -> i32;
    pub fn diffusion_gemma_commit(
        model: *mut std::ffi::c_void,
        tokens: *const i32,
        len: i32,
    ) -> i32;
    pub fn diffusion_gemma_predict_canvas(
        model: *mut std::ffi::c_void,
        canvas: *const i32,
        canvas_len: i32,
        valid_len: i32,
        step: i32,
        temperature: f32,
        out_sampled: *mut u32,
        out_argmax: *mut u32,
        out_entropy: *mut f32,
    ) -> i32;
    pub fn diffusion_gemma_generate(
        model: *mut std::ffi::c_void,
        prompt: *const i32,
        prompt_len: i32,
        max_new_tokens: i32,
        canvas_len: i32,
        max_steps: i32,
        entropy_bound: f32,
        confidence_threshold: f32,
        t_min: f32,
        t_max: f32,
        stability_threshold: i32,
        seed: u64,
        stop_ids: *const u32,
        stop_ids_len: i32,
        cancel_fn: Option<unsafe extern "C" fn(ctx: *const std::ffi::c_void) -> i32>,
        cancel_ctx: *const std::ffi::c_void,
        out_tokens: *mut u32,
        out_len: *mut i32,
        out_finish: *mut i32,
        out_blocks: *mut i32,
        out_steps: *mut i32,
        out_forced: *mut i32,
        out_adaptive: *mut i32,
    ) -> i32;
    pub fn diffusion_gemma_generate_causal(
        model: *mut std::ffi::c_void,
        prompt: *const i32,
        prompt_len: i32,
        max_new_tokens: i32,
        seed: u64,
        stop_ids: *const u32,
        stop_ids_len: i32,
        cancel_fn: Option<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
        cancel_ctx: *const std::ffi::c_void,
        out_tokens: *mut u32,
        out_len: *mut i32,
        out_finish: *mut i32,
    ) -> i32;
    pub fn diffusion_gemma_generate_causal_image(
        model: *mut std::ffi::c_void,
        prompt: *const i32,
        prompt_len: i32,
        image_pixels: *const f32,
        image_height: i32,
        image_width: i32,
        image_soft_tokens: i32,
        max_new_tokens: i32,
        seed: u64,
        stop_ids: *const u32,
        stop_ids_len: i32,
        cancel_fn: Option<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
        cancel_ctx: *const std::ffi::c_void,
        out_tokens: *mut u32,
        out_len: *mut i32,
        out_finish: *mut i32,
    ) -> i32;

    //
    // VLM with a DeepEncoder (SAM-base + CLIP-large + 16x conv compressor +
    // linear projector) and a DeepSeek-MoE decoder. Decoder weights are MXFP8
    // (scales uint8, no biases); vision weights are dense BF16. Weight ids index
    // into one registry; `add_*` returns the id, layer-push references them.

    pub fn deepseek_ocr_new() -> *mut std::ffi::c_void;
    pub fn deepseek_ocr_free(model: *mut std::ffi::c_void);
    pub fn deepseek_ocr_add_dense_weight(model: *mut std::ffi::c_void, w: *mut mlx_array) -> i32;
    /// Register an MXFP8 quantized weight (uint8 scales, no biases).
    pub fn deepseek_ocr_add_mxfp8_weight(
        model: *mut std::ffi::c_void,
        w: *mut mlx_array,
        scales: *mut mlx_array,
        group_size: i32,
        bits: i32,
    ) -> i32;
    pub fn deepseek_ocr_set_config(
        model: *mut std::ffi::c_void,
        hidden_size: i32,
        vocab_size: i32,
        num_attention_heads: i32,
        num_key_value_heads: i32,
        head_dim: i32,
        v_head_dim: i32,
        rms_norm_eps: f32,
        rope_theta: f32,
    );
    pub fn deepseek_ocr_set_embed(
        model: *mut std::ffi::c_void,
        embed_id: i32,
        embed_scales_id: i32,
        lm_head_id: i32,
        lm_head_scales_id: i32,
        final_norm_id: i32,
        quant_group_size: i32,
        quant_bits: i32,
    );
    /// Push one decoder layer. Dense layer: pass `num_experts<=0` and use the
    /// `dense_*` projection ids; MoE layer: pass `num_experts>0` with router +
    /// stacked switch-expert + fused shared-expert ids.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek_ocr_push_layer(
        model: *mut std::ffi::c_void,
        input_ln_id: i32,
        post_attn_ln_id: i32,
        q_id: i32,
        k_id: i32,
        v_id: i32,
        o_id: i32,
        // dense MLP (used when num_experts <= 0)
        dense_gate_id: i32,
        dense_up_id: i32,
        dense_down_id: i32,
        // MoE (used when num_experts > 0)
        router_id: i32,
        switch_gate_id: i32,
        switch_up_id: i32,
        switch_down_id: i32,
        shared_gate_id: i32,
        shared_up_id: i32,
        shared_down_id: i32,
        num_experts: i32,
        top_k: i32,
        routed_scaling_factor: f32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek_ocr_set_vision_config(
        model: *mut std::ffi::c_void,
        image_token_id: i32,
        // CLIP
        clip_hidden_size: i32,
        clip_intermediate_size: i32,
        clip_num_layers: i32,
        clip_num_heads: i32,
        clip_patch_size: i32,
        clip_layer_norm_eps: f32,
        // SAM
        sam_width: i32,
        sam_layers: i32,
        sam_heads: i32,
        sam_patch_size: i32,
        sam_window_size: i32,
        sam_image_size: i32,
        // projector
        projector_input_dim: i32,
        projector_n_embed: i32,
    ) -> i32;
    /// SAM patch_embed (conv weight [out,kh,kw,in] + bias), absolute pos_embed,
    /// neck convs (0/2 weights, 1/3 layernorm w+b), net_2/net_3 convs.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek_ocr_set_sam_stem(
        model: *mut std::ffi::c_void,
        patch_embed_w_id: i32,
        patch_embed_b_id: i32,
        pos_embed_id: i32,
        neck0_w_id: i32,
        neck1_w_id: i32,
        neck1_b_id: i32,
        neck2_w_id: i32,
        neck3_w_id: i32,
        neck3_b_id: i32,
        net2_w_id: i32,
        net3_w_id: i32,
    ) -> i32;
    /// One SAM ViT block. `window_size=0` marks a global-attention block.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek_ocr_push_sam_block(
        model: *mut std::ffi::c_void,
        window_size: i32,
        norm1_w_id: i32,
        norm1_b_id: i32,
        qkv_w_id: i32,
        qkv_b_id: i32,
        proj_w_id: i32,
        proj_b_id: i32,
        rel_pos_h_id: i32,
        rel_pos_w_id: i32,
        norm2_w_id: i32,
        norm2_b_id: i32,
        lin1_w_id: i32,
        lin1_b_id: i32,
        lin2_w_id: i32,
        lin2_b_id: i32,
    ) -> i32;
    /// CLIP embeddings + pre-layernorm stem.
    pub fn deepseek_ocr_set_clip_stem(
        model: *mut std::ffi::c_void,
        class_embedding_id: i32,
        position_embedding_id: i32,
        pre_layernorm_w_id: i32,
        pre_layernorm_b_id: i32,
    ) -> i32;
    /// One CLIP encoder layer (fused qkv, layernorm w+b, MLP fc1/fc2).
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek_ocr_push_clip_layer(
        model: *mut std::ffi::c_void,
        ln1_w_id: i32,
        ln1_b_id: i32,
        qkv_w_id: i32,
        qkv_b_id: i32,
        out_w_id: i32,
        out_b_id: i32,
        ln2_w_id: i32,
        ln2_b_id: i32,
        fc1_w_id: i32,
        fc1_b_id: i32,
        fc2_w_id: i32,
        fc2_b_id: i32,
    ) -> i32;
    /// Projector (MXFP8 linear w+scales+bias) + image_newline + view_separator.
    pub fn deepseek_ocr_set_projector(
        model: *mut std::ffi::c_void,
        projector_w_id: i32,
        projector_bias_id: i32,
        image_newline_id: i32,
        view_separator_id: i32,
    ) -> i32;
    pub fn deepseek_ocr_finalize(model: *mut std::ffi::c_void) -> i32;
    pub fn deepseek_ocr_generate_causal(
        model: *mut std::ffi::c_void,
        prompt: *const i32,
        prompt_len: i32,
        max_new_tokens: i32,
        seed: u64,
        stop_ids: *const u32,
        stop_ids_len: i32,
        cancel_fn: Option<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
        cancel_ctx: *const std::ffi::c_void,
        out_tokens: *mut u32,
        out_len: *mut i32,
        out_finish: *mut i32,
    ) -> i32;
    pub fn deepseek_ocr_generate_causal_image(
        model: *mut std::ffi::c_void,
        prompt: *const i32,
        prompt_len: i32,
        image_pixels: *const f32,
        image_height: i32,
        image_width: i32,
        image_soft_tokens: i32,
        max_new_tokens: i32,
        seed: u64,
        stop_ids: *const u32,
        stop_ids_len: i32,
        cancel_fn: Option<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
        cancel_ctx: *const std::ffi::c_void,
        out_tokens: *mut u32,
        out_len: *mut i32,
        out_finish: *mut i32,
    ) -> i32;

    pub fn qwen35_compiled_new() -> *mut std::ffi::c_void;
    pub fn qwen35_compiled_free(model: *mut std::ffi::c_void);
    pub fn qwen35_compiled_add_dense_weight(model: *mut std::ffi::c_void, w: *mut mlx_array)
    -> i32;
    pub fn qwen35_compiled_add_quant_weight(
        model: *mut std::ffi::c_void,
        w: *mut mlx_array,
        scales: *mut mlx_array,
        biases: *mut mlx_array,
        group_size: i32,
        bits: i32,
        mode: i32,
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
    pub fn qwen35_compiled_set_embed_v2(
        model: *mut std::ffi::c_void,
        embed_tokens: *mut mlx_array,
        final_norm_w: *mut mlx_array,
        lm_head_id: i32,
    );
    pub fn qwen35_compiled_set_embed_as_linear_v2(model: *mut std::ffi::c_void, embed_id: i32);
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
    pub fn qwen35_compiled_set_separate_mlp_v2(
        model: *mut std::ffi::c_void,
        gate_id: i32,
        up_id: i32,
    );
    pub fn qwen35_compiled_set_full_separate_mlp_v2(
        model: *mut std::ffi::c_void,
        gate_id: i32,
        up_id: i32,
    );
    pub fn qwen35_compiled_finalize(model: *mut std::ffi::c_void) -> i32;
    #[allow(clippy::too_many_arguments)]
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
    /// Paged single-step session decode. BF16 sessions accept pre-gathered
    /// per-layer prefix K/V tensors. INT8 sessions accept flat per-layer
    /// q/scale/bias triples for K and V. The session still writes the fresh
    /// token into its contiguous cache for compatibility. Passing zero layers is
    /// a legacy fallthrough; batch/verify paths keep separate entrypoints.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_compiled_step_session_paged(
        model: *mut std::ffi::c_void,
        token_id: *mut mlx_array,
        cache_pos: i32,
        k_full_per_layer: *mut *mut mlx_array,
        v_full_per_layer: *mut *mut mlx_array,
        n_full_layers: i32,
        k_int8_full_per_layer: *mut *mut mlx_array,
        v_int8_full_per_layer: *mut *mut mlx_array,
        n_int8_full_layers: i32,
        out_logits: *mut *mut mlx_array,
    ) -> i32;
    pub fn qwen35_compiled_prefill_session(
        model: *mut std::ffi::c_void,
        token_ids: *mut mlx_array,
        prompt_len: i32,
        cache_pos: i32,
        out_logits: *mut *mut mlx_array,
    ) -> i32;
    /// DFlash verify: parallel forward over a draft block, returning all-position
    /// logits [1, block_size, vocab]. Respects model-level tape_mode and capture
    /// layers — one call emits per-step GDR tapes and captured hidden for the
    /// entire block, replacing the previous 16 × seq_len=1 sequential verify loop.
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
        accept_topk: i32,
        out_matched_prefix_len: *mut i32,
        out_next_token: *mut i32,
        out_kv_caches: *mut *mut mlx_array,
        out_gdr_states: *mut *mut mlx_array,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]

    /// Full decode loop in C++ — all intermediates stay alive within the loop.
    #[allow(clippy::too_many_arguments)]

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
    /// 2-D quantized linears (affine scale+bias; the compiled MLP fuses them).
    ///
    /// Returns a newly-allocated array handle (caller must `mlx_array_free`)
    /// or nullptr on failure (check `mlx_last_error()`).
    #[allow(clippy::too_many_arguments)]

    /// RMS normalization. Pass null for weight to use no learnable weight.
    pub fn mlx_fast_rms_norm(x: *mut mlx_array, weight: *mut mlx_array, eps: f32)
    -> *mut mlx_array;
    #[allow(clippy::too_many_arguments)]
    pub fn mlx_tape_replay(
        tape: *mut mlx_array,
        k: *mut mlx_array,
        g: *mut mlx_array,
        state_in: *mut mlx_array,
        steps: i32,
    ) -> *mut mlx_array;

    pub fn qwen35_set_tape_mode(model: *mut std::ffi::c_void, enabled: bool);
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

    pub fn mlx_eval(arrays: *mut *mut mlx_array, count: usize);
    pub fn mlx_async_eval(arrays: *mut *mut mlx_array, count: usize);

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
