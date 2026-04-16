#include "mlx_common.h"

extern "C" {

// === Error handling ===

const char* mlx_last_error() {
    return g_mlx_last_error.empty() ? nullptr : g_mlx_last_error.c_str();
}

// === Array lifecycle ===

mlx_array* mlx_array_new_float32(float val) {
    MLX_TRY_RETURN(from_arr(array(val)));
}

mlx_array* mlx_array_new_int32(int32_t val) {
    MLX_TRY_RETURN(from_arr(array(val)));
}

mlx_array* mlx_array_from_data(const void* data, const int32_t* shape, int32_t ndim, int32_t dtype_val) {
    MLX_TRY_RETURN([&]() {
        auto sh = make_shape(shape, static_cast<size_t>(ndim));
        auto dt = to_dtype(dtype_val);
        // MLX array constructor needs the allocator to copy data
        size_t nbytes = 1;
        for (int i = 0; i < ndim; i++) nbytes *= shape[i];
        nbytes *= size_of(dt);
        auto buf = allocator::malloc(nbytes);
        std::memcpy(buf.raw_ptr(), data, nbytes);
        return reinterpret_cast<mlx_array*>(new array(std::move(buf), sh, dt));
    }());
}

mlx_array* mlx_array_clone(mlx_array* a) {
    // Copy the shared_ptr (increment refcount, same underlying data)
    MLX_TRY_RETURN(reinterpret_cast<mlx_array*>(new array(*to_arr(a))));
}

void mlx_array_free(mlx_array* a) {
    MLX_TRY_VOID(delete to_arr(a));
}

int32_t mlx_array_ndim(mlx_array* a) {
    MLX_TRY_RETURN_VALUE(0, static_cast<int32_t>(to_arr(a)->ndim()));
}

const int32_t* mlx_array_shape(mlx_array* a) {
    // MLX shape() returns std::vector<int>; data() gives stable pointer
    // while the array is alive.
    MLX_TRY_RETURN(to_arr(a)->shape().data());
}

int32_t mlx_array_dtype(mlx_array* a) {
    MLX_TRY_RETURN_VALUE(10 /*float32 fallback*/, from_dtype(to_arr(a)->dtype()));
}

int32_t mlx_array_item_int32(mlx_array* a) {
    MLX_TRY_RETURN_VALUE(0, to_arr(a)->item<int32_t>());
}

float mlx_array_item_float32(mlx_array* a) {
    MLX_TRY_RETURN_VALUE(0.0f, to_arr(a)->item<float>());
}

const float* mlx_array_data_float32(mlx_array* a) {
    MLX_TRY_RETURN(to_arr(a)->data<float>());
}

const int32_t* mlx_array_data_int32(mlx_array* a) {
    MLX_TRY_RETURN(to_arr(a)->data<int32_t>());
}

size_t mlx_array_size(mlx_array* a) {
    MLX_TRY_RETURN_VALUE(0, to_arr(a)->size());
}

// === Binary ops ===

mlx_array* mlx_add(mlx_array* a, mlx_array* b) {
    MLX_TRY_RETURN(from_arr(add(*to_arr(a), *to_arr(b))));
}

mlx_array* mlx_subtract(mlx_array* a, mlx_array* b) {
    MLX_TRY_RETURN(from_arr(subtract(*to_arr(a), *to_arr(b))));
}

mlx_array* mlx_multiply(mlx_array* a, mlx_array* b) {
    MLX_TRY_RETURN(from_arr(multiply(*to_arr(a), *to_arr(b))));
}

mlx_array* mlx_matmul(mlx_array* a, mlx_array* b) {
    MLX_TRY_RETURN(from_arr(matmul(*to_arr(a), *to_arr(b))));
}

mlx_array* mlx_greater(mlx_array* a, mlx_array* b) {
    MLX_TRY_RETURN(from_arr(greater(*to_arr(a), *to_arr(b))));
}

// === Unary ops ===

mlx_array* mlx_exp(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(exp(*to_arr(a))));
}

mlx_array* mlx_log1p(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(log1p(*to_arr(a))));
}

mlx_array* mlx_negative(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(negative(*to_arr(a))));
}

mlx_array* mlx_sqrt(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(sqrt(*to_arr(a))));
}

mlx_array* mlx_reciprocal(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(reciprocal(*to_arr(a))));
}

mlx_array* mlx_sigmoid(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(sigmoid(*to_arr(a))));
}

// === Shape ops ===

mlx_array* mlx_reshape(mlx_array* a, const int32_t* shape, size_t ndim) {
    MLX_TRY_RETURN([&]() {
        auto sh = make_shape(shape, ndim);
        return from_arr(reshape(*to_arr(a), sh));
    }());
}

mlx_array* mlx_transpose(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(transpose(*to_arr(a))));
}

mlx_array* mlx_transpose_axes(mlx_array* a, const int32_t* axes, size_t n) {
    MLX_TRY_RETURN([&]() {
        std::vector<int> ax(axes, axes + n);
        return from_arr(transpose(*to_arr(a), ax));
    }());
}

mlx_array* mlx_astype(mlx_array* a, int32_t dtype) {
    MLX_TRY_RETURN(from_arr(astype(*to_arr(a), to_dtype(dtype))));
}

mlx_array* mlx_broadcast_to(mlx_array* a, const int32_t* shape, size_t ndim) {
    MLX_TRY_RETURN([&]() {
        auto sh = make_shape(shape, ndim);
        return from_arr(broadcast_to(*to_arr(a), sh));
    }());
}

mlx_array* mlx_expand_dims(mlx_array* a, int32_t axis) {
    MLX_TRY_RETURN(from_arr(expand_dims(*to_arr(a), static_cast<int>(axis))));
}

mlx_array* mlx_zeros(const int32_t* shape, size_t ndim, int32_t dtype) {
    MLX_TRY_RETURN([&]() {
        auto sh = make_shape(shape, ndim);
        return from_arr(zeros(sh, to_dtype(dtype)));
    }());
}

// === Indexing ===

mlx_array* mlx_take_axis(mlx_array* a, mlx_array* indices, int32_t axis) {
    MLX_TRY_RETURN(from_arr(take(*to_arr(a), *to_arr(indices), static_cast<int>(axis))));
}

mlx_array* mlx_slice(mlx_array* a, const int32_t* start, const int32_t* stop,
                     const int32_t* strides, size_t ndim) {
    MLX_TRY_RETURN([&]() {
        Shape st; for(size_t i=0;i<ndim;i++) st.push_back(start[i]);
        Shape sp; for(size_t i=0;i<ndim;i++) sp.push_back(stop[i]);
        Shape sr; for(size_t i=0;i<ndim;i++) sr.push_back(strides[i]);
        return from_arr(slice(*to_arr(a), st, sp, sr));
    }());
}

mlx_array* mlx_slice_update(mlx_array* src, mlx_array* update,
                            const int32_t* start, const int32_t* stop,
                            const int32_t* strides, size_t ndim) {
    MLX_TRY_RETURN([&]() {
        Shape st; for(size_t i=0;i<ndim;i++) st.push_back(start[i]);
        Shape sp; for(size_t i=0;i<ndim;i++) sp.push_back(stop[i]);
        Shape sr; for(size_t i=0;i<ndim;i++) sr.push_back(strides[i]);
        return from_arr(slice_update(*to_arr(src), *to_arr(update), st, sp, sr));
    }());
}

mlx_array* mlx_concatenate_axis(mlx_array** arrays, size_t count, int32_t axis) {
    MLX_TRY_RETURN([&]() {
        std::vector<array> arrs;
        arrs.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            arrs.push_back(*to_arr(arrays[i]));
        }
        return from_arr(concatenate(arrs, static_cast<int>(axis)));
    }());
}

mlx_array* mlx_where(mlx_array* cond, mlx_array* a, mlx_array* b) {
    MLX_TRY_RETURN(from_arr(where(*to_arr(cond), *to_arr(a), *to_arr(b))));
}

// === Reduction ===

mlx_array* mlx_sum_axis(mlx_array* a, int32_t axis, bool keepdims) {
    MLX_TRY_RETURN(from_arr(sum(*to_arr(a), static_cast<int>(axis), keepdims)));
}

mlx_array* mlx_argmax(mlx_array* a, bool keepdims) {
    MLX_TRY_RETURN(from_arr(argmax(*to_arr(a), keepdims)));
}

mlx_array* mlx_argmax_axis(mlx_array* a, int axis, bool keepdims) {
    MLX_TRY_RETURN(from_arr(argmax(*to_arr(a), axis, keepdims)));
}

// === Quantized ===

mlx_array* mlx_quantized_matmul(mlx_array* x, mlx_array* w, mlx_array* scales,
                                mlx_array* biases, bool transpose,
                                int32_t group_size, int32_t bits) {
    MLX_TRY_RETURN(from_arr(quantized_matmul(
        *to_arr(x), *to_arr(w), *to_arr(scales), *to_arr(biases),
        transpose, group_size, bits)));
}

mlx_array* mlx_dequantize(mlx_array* w, mlx_array* scales, mlx_array* biases,
                          int32_t group_size, int32_t bits) {
    MLX_TRY_RETURN(from_arr(dequantize(*to_arr(w), *to_arr(scales), *to_arr(biases),
                                       group_size, bits)));
}

// === Fast ops ===

mlx_array* mlx_fast_rms_norm(mlx_array* x, mlx_array* weight, float eps) {
    MLX_TRY_RETURN([&]() {
        if (weight == nullptr) {
            return from_arr(fast::rms_norm(*to_arr(x), std::nullopt, eps));
        }
        return from_arr(fast::rms_norm(*to_arr(x), *to_arr(weight), eps));
    }());
}

mlx_array* mlx_fast_rope(mlx_array* x, int32_t dims, bool traditional,
                         float base, float scale, int32_t offset) {
    MLX_TRY_RETURN(from_arr(fast::rope(
        *to_arr(x), dims, traditional, base, scale, offset)));
}

mlx_array* mlx_fast_rope_dynamic(mlx_array* x, int32_t dims, bool traditional,
                                 float base, float scale, mlx_array* offset) {
    MLX_TRY_RETURN(from_arr(fast::rope(
        *to_arr(x), dims, traditional, base, scale, *to_arr(offset))));
}

mlx_array* mlx_fast_sdpa(mlx_array* q, mlx_array* k, mlx_array* v,
                         float scale, const char* mask_mode) {
    // mask_mode: "" for no mask, "causal" for causal masking.
    // NEVER pass nullptr — std::string(nullptr) is UB.
    MLX_TRY_RETURN([&]() {
        std::string mode(mask_mode);
        return from_arr(fast::scaled_dot_product_attention(
            *to_arr(q), *to_arr(k), *to_arr(v), scale, mode));
    }());
}

mlx_array* mlx_fast_sdpa_masked(mlx_array* q, mlx_array* k, mlx_array* v,
                                float scale, mlx_array* mask_arr) {
    MLX_TRY_RETURN([&]() {
        return from_arr(fast::scaled_dot_product_attention(
            *to_arr(q), *to_arr(k), *to_arr(v), scale, "",
            *to_arr(mask_arr)));
    }());
}

// === Random ===

mlx_array* mlx_random_categorical(mlx_array* logits, int32_t axis) {
    MLX_TRY_RETURN(from_arr(random::categorical(*to_arr(logits), static_cast<int>(axis))));
}

// === Transforms ===

void mlx_eval(mlx_array** arrays, size_t count) {
    try {
        mlx_clear_error();
        std::vector<array> arrs;
        arrs.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            arrs.push_back(*to_arr(arrays[i]));
        }
        eval(arrs);
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
    }
}

void mlx_async_eval(mlx_array** arrays, size_t count) {
    try {
        mlx_clear_error();
        std::vector<array> arrs;
        arrs.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            arrs.push_back(*to_arr(arrays[i]));
        }
        async_eval(arrs);
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
    }
}

// === IO ===

int32_t mlx_load_safetensors(const char* path,
                             const char*** out_names,
                             mlx_array*** out_arrays) {
    mlx_clear_error();
    std::pair<std::unordered_map<std::string, array>, std::unordered_map<std::string, std::string>> result;
    try {
        result = load_safetensors(std::string(path));
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        *out_names = nullptr;
        *out_arrays = nullptr;
        return -1;
    }
    auto& data = result.first;
    int32_t count = static_cast<int32_t>(data.size());
    if (count == 0) {
        *out_names = nullptr;
        *out_arrays = nullptr;
        return 0;
    }

    auto** names = new const char*[count];
    auto** arrays = new mlx_array*[count];
    int32_t i = 0;
    for (auto& [key, val] : data) {
        // Duplicate the string so Rust can free it later
        char* name = new char[key.size() + 1];
        std::memcpy(name, key.c_str(), key.size() + 1);
        names[i] = name;
        arrays[i] = from_arr(std::move(val));
        ++i;
    }
    *out_names = names;
    *out_arrays = arrays;
    return count;
}

void mlx_free_loaded_tensors(const char** names, mlx_array** arrays, int32_t count) {
    for (int32_t i = 0; i < count; ++i) {
        delete[] names[i];
        delete to_arr(arrays[i]);
    }
    delete[] names;
    delete[] arrays;
}

// === Contiguous ===

mlx_array* mlx_contiguous(mlx_array* a) {
    MLX_TRY_RETURN(from_arr(contiguous(*to_arr(a))));
}

// === Conv1d ===

mlx_array* mlx_conv1d(mlx_array* input, mlx_array* weight,
                      int32_t stride, int32_t padding,
                      int32_t dilation, int32_t groups) {
    MLX_TRY_RETURN(from_arr(conv1d(*to_arr(input), *to_arr(weight),
                                   stride, padding, dilation, groups)));
}

// === Fused compute_g ===
// g = exp(-exp(A_log) * softplus(alpha + dt_bias))
// Fuses 10 ops into 1 C++ call (no heap allocs for intermediates).

mlx_array* mlx_compute_g(mlx_array* A_log, mlx_array* alpha, mlx_array* dt_bias) {
    MLX_TRY_RETURN([&]() {
        auto a = astype(*to_arr(A_log), float32);
        auto ab = *to_arr(alpha) + *to_arr(dt_bias);
        auto sp = where(greater(ab, array(20.0f)), ab, log1p(exp(ab)));
        return from_arr(exp(negative(exp(a)) * sp));
    }());
}

// Metal kernel struct (used by both GDR forward and standalone kernel API)
struct mlx_metal_kernel {
    std::string name;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string source;
    std::string header;
};

// === Fused GDR layer forward ===
// Full GDR linear attention decode step in C++ — eliminates ~40 FFI calls per layer.
// Matches mlx_lm's GatedDeltaNet.__call__ for decode (T=1).

void mlx_gdr_layer_forward(
    mlx_array* x_in,           // [1, 1, hidden] bf16 input
    // Projection weights (quantized)
    mlx_array* qkvz_w, mlx_array* qkvz_s, mlx_array* qkvz_b,
    int32_t qkvz_gs, int32_t qkvz_bits,
    int32_t qkv_split, int32_t z_split,
    mlx_array* ba_w, mlx_array* ba_s, mlx_array* ba_b,
    int32_t ba_gs, int32_t ba_bits, int32_t ba_num_heads,
    // Conv1d
    mlx_array* conv1d_w,       // [C, K, 1] bf16
    mlx_array** conv_state,    // [1, K-1, C] bf16 — updated in place
    int32_t conv_kernel,
    // Gate params
    mlx_array* a_log,          // [Hv] f32
    mlx_array* dt_bias,        // [Hv] bf16
    // Norm + output
    mlx_array* norm_w,         // [Dv] bf16
    float rms_eps,
    mlx_array* out_w, mlx_array* out_s, mlx_array* out_b,
    int32_t out_gs, int32_t out_bits,
    // Config
    int32_t num_key_heads, int32_t key_dim,
    int32_t num_value_heads, int32_t value_dim,
    float q_scale, float k_scale,
    // State
    mlx_array** gdr_state,     // [1, Hv, Dv, Dk] f32 — updated in place
    // Metal kernel handle
    void* metal_kernel, int32_t use_metal_kernel,
    // Output
    mlx_array* out_result      // receives the output array
) {
    try {
        mlx_clear_error();
        auto x = *to_arr(x_in);
        int hk = num_key_heads, dk = key_dim;
        int hv = num_value_heads, dv = value_dim;
        int q_dim = hk * dk, k_dim_total = q_dim, v_dim = hv * dv;
        int qkv_dim = q_dim + k_dim_total + v_dim;

        // 1. Projections
        auto qkvz = quantized_matmul(x, *to_arr(qkvz_w), *to_arr(qkvz_s), *to_arr(qkvz_b),
                                     true, qkvz_gs, qkvz_bits);
        auto qkv = slice(qkvz, {0, 0, 0}, {1, 1, qkv_split});
        auto z = slice(qkvz, {0, 0, qkv_split}, {1, 1, qkv_split + z_split});

        auto ba = quantized_matmul(x, *to_arr(ba_w), *to_arr(ba_s), *to_arr(ba_b),
                                   true, ba_gs, ba_bits);
        auto b_raw = slice(ba, {0, 0, 0}, {1, 1, ba_num_heads});
        auto a_raw = slice(ba, {0, 0, ba_num_heads}, {1, 1, ba_num_heads * 2});

        // 2. Conv1d (standard depthwise)
        auto conv_st = *to_arr(*conv_state);
        auto conv_input = concatenate({conv_st, qkv}, 1);
        int n_keep = conv_kernel - 1;
        auto new_conv_state = slice(conv_input, {0, 1, 0}, {1, n_keep + 1, qkv_dim});
        delete to_arr(*conv_state);
        *conv_state = from_arr(std::move(new_conv_state));

        auto conv_out = conv1d(conv_input, *to_arr(conv1d_w), 1, 0, 1, qkv_dim);
        conv_out = conv_out * sigmoid(conv_out);

        // 3. Split QKV + RMS normalize
        auto q_raw = reshape(slice(conv_out, {0, 0, 0}, {1, 1, q_dim}), {1, 1, hk, dk});
        auto k_raw = reshape(slice(conv_out, {0, 0, q_dim}, {1, 1, q_dim + k_dim_total}), {1, 1, hk, dk});
        auto v_raw = reshape(
            slice(conv_out, {0, 0, q_dim + k_dim_total}, {1, 1, q_dim + k_dim_total + v_dim}),
            {1, 1, hv, dv});

        auto q = fast::rms_norm(q_raw, std::nullopt, 1e-6f) * array(q_scale);
        auto k = fast::rms_norm(k_raw, std::nullopt, 1e-6f) * array(k_scale);

        // 4. Gate computation
        auto beta = sigmoid(b_raw);
        auto A = *to_arr(a_log);
        auto ab = a_raw + *to_arr(dt_bias);
        auto sp = where(greater(ab, array(20.0f)), ab, log1p(exp(ab)));
        auto g = exp(negative(exp(astype(A, float32))) * sp);

        // 5. Metal kernel state update
        array y_out({0});
        if (use_metal_kernel && metal_kernel) {
            auto q_bf16 = astype(q, bfloat16);
            auto k_bf16 = astype(k, bfloat16);
            auto v_bf16 = astype(v_raw, bfloat16);
            auto g_3d = reshape(g, {1, 1, hv});
            auto beta_3d = reshape(beta, {1, 1, hv});
            auto state_in = *to_arr(*gdr_state);

            auto* mk = static_cast<mlx_metal_kernel*>(metal_kernel);
            auto kernel_fn = fast::metal_kernel(
                mk->name, mk->input_names, mk->output_names,
                mk->source, mk->header, true, false);

            auto t_arr = array(1);
            std::vector<array> inputs = {q_bf16, k_bf16, v_bf16, g_3d, beta_3d, state_in, t_arr};
            std::vector<Shape> out_shapes = {{1, 1, hv, dv}, state_in.shape()};
            std::vector<Dtype> out_dtypes = {bfloat16, float32};
            std::vector<std::pair<std::string, fast::TemplateArg>> tmpl = {
                {"Dk", fast::TemplateArg(dk)}, {"Dv", fast::TemplateArg(dv)},
                {"Hk", fast::TemplateArg(hk)}, {"Hv", fast::TemplateArg(hv)},
                {"InT", fast::TemplateArg(bfloat16)}, {"StT", fast::TemplateArg(float32)}
            };

            auto result = kernel_fn(
                inputs, out_shapes, out_dtypes,
                std::make_tuple(32, dv, 1 * hv),
                std::make_tuple(32, 4, 1),
                tmpl, std::nullopt, false, {});

            y_out = result[0];
            delete to_arr(*gdr_state);
            *gdr_state = from_arr(std::move(result[1]));
        } else {
            auto g_4d = reshape(g, {1, hv, 1, 1});
            auto s = *to_arr(*gdr_state);
            auto s_decayed = s * g_4d;
            int heads_per_key = hv / hk;
            array k_exp = (heads_per_key > 1)
                ? reshape(broadcast_to(expand_dims(k, 2), {1, 1, hk, heads_per_key, dk}), {1, hv, dk})
                : reshape(k, {1, hv, dk});
            array q_exp = (heads_per_key > 1)
                ? reshape(broadcast_to(expand_dims(q, 2), {1, 1, hk, heads_per_key, dk}), {1, hv, dk})
                : reshape(q, {1, hv, dk});
            auto v_3d = reshape(v_raw, {1, hv, dv});
            auto k_4d = reshape(k_exp, {1, hv, 1, dk});
            auto kv_mem = sum(s_decayed * k_4d, -1, false);
            auto beta_3d = reshape(beta, {1, hv, 1});
            auto delta = (v_3d - kv_mem) * beta_3d;
            auto s_updated = s_decayed + reshape(delta, {1, hv, dv, 1}) * k_4d;
            auto q_4d = reshape(q_exp, {1, hv, 1, dk});
            y_out = reshape(sum(s_updated * q_4d, -1, false), {1, 1, hv, dv});
            delete to_arr(*gdr_state);
            *gdr_state = from_arr(std::move(s_updated));
        }

        // 6. Per-head RMSNorm + output gate
        auto y_heads = reshape(y_out, {hv, dv});
        auto normed = fast::rms_norm(y_heads, *to_arr(norm_w), rms_eps);
        auto z_gated = reshape(z, {hv, dv});
        auto silu_z = z_gated * sigmoid(z_gated);
        auto gated_out = normed * silu_z;

        // 7. Output projection
        auto out_flat = reshape(gated_out, {1, hv * dv});
        auto result = quantized_matmul(out_flat, *to_arr(out_w), *to_arr(out_s), *to_arr(out_b),
                                       true, out_gs, out_bits);

        auto** dst = reinterpret_cast<mlx_array**>(out_result);
        *dst = from_arr(std::move(result));
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
        auto** dst = reinterpret_cast<mlx_array**>(out_result);
        *dst = nullptr;
    }
}

// === Metal kernel ===

void* mlx_metal_kernel_new(const char* name,
                           const char** input_names, size_t n_inputs,
                           const char** output_names, size_t n_outputs,
                           const char* source, const char* header) {
    MLX_TRY_RETURN_VALUE(nullptr, [&]() -> void* {
        auto* k = new mlx_metal_kernel();
        k->name = std::string(name);
        for (size_t i = 0; i < n_inputs; ++i) {
            k->input_names.emplace_back(input_names[i]);
        }
        for (size_t i = 0; i < n_outputs; ++i) {
            k->output_names.emplace_back(output_names[i]);
        }
        k->source = std::string(source);
        k->header = std::string(header ? header : "");
        return k;
    }());
}

void mlx_metal_kernel_free(void* kernel) {
    MLX_TRY_VOID(delete static_cast<mlx_metal_kernel*>(kernel));
}

void mlx_metal_kernel_apply(void* kernel,
                            mlx_array** inputs, size_t n_inputs,
                            mlx_array** outputs, size_t n_outputs,
                            const int32_t* grid, const int32_t* threadgroup,
                            const int32_t* output_shapes,
                            const int32_t* output_shape_dims,
                            const int32_t* output_dtypes,
                            const char** template_names,
                            const int32_t* template_int_vals,
                            const int32_t* template_dtype_vals,
                            size_t n_int_templates,
                            size_t n_dtype_templates) {
    try {
        mlx_clear_error();
        auto* k = static_cast<mlx_metal_kernel*>(kernel);

        std::vector<array> in_arrs;
        in_arrs.reserve(n_inputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            in_arrs.push_back(*to_arr(inputs[i]));
        }

        std::vector<Shape> out_shapes;
        std::vector<Dtype> out_dtypes;
        int shape_offset = 0;
        for (size_t i = 0; i < n_outputs; ++i) {
            int ndim = output_shape_dims[i];
            Shape sh(output_shapes + shape_offset, output_shapes + shape_offset + ndim);
            out_shapes.push_back(sh);
            out_dtypes.push_back(to_dtype(output_dtypes[i]));
            shape_offset += ndim;
        }

        std::vector<std::pair<std::string, fast::TemplateArg>> template_args;
        for (size_t i = 0; i < n_int_templates; ++i) {
            template_args.emplace_back(
                std::string(template_names[i]),
                fast::TemplateArg(template_int_vals[i]));
        }
        for (size_t i = 0; i < n_dtype_templates; ++i) {
            template_args.emplace_back(
                std::string(template_names[n_int_templates + i]),
                fast::TemplateArg(to_dtype(template_dtype_vals[i])));
        }

        auto mk = fast::metal_kernel(
            k->name,
            k->input_names,
            k->output_names,
            k->source,
            k->header,
            true,
            false
        );

        auto grid_tuple = std::make_tuple(grid[0], grid[1], grid[2]);
        auto tg_tuple = std::make_tuple(threadgroup[0], threadgroup[1], threadgroup[2]);

        auto result = mk(
            in_arrs, out_shapes, out_dtypes,
            grid_tuple, tg_tuple,
            template_args,
            std::nullopt,
            false,
            {}
        );

        for (size_t i = 0; i < result.size() && i < n_outputs; ++i) {
            outputs[i] = from_arr(std::move(result[i]));
        }
    } catch (const std::exception& e) {
        mlx_set_error(e.what());
    }
}

// === Memory management ===

/// Current active MLX allocator memory in bytes.
size_t mlx_get_active_memory() {
    MLX_TRY_RETURN_VALUE(0, mlx::core::get_active_memory());
}

/// Peak MLX allocator memory in bytes.
size_t mlx_get_peak_memory() {
    MLX_TRY_RETURN_VALUE(0, mlx::core::get_peak_memory());
}

/// Cached MLX allocator memory in bytes.
size_t mlx_get_cache_memory() {
    MLX_TRY_RETURN_VALUE(0, mlx::core::get_cache_memory());
}

/// Set the MLX allocator memory limit in bytes.
size_t mlx_set_memory_limit(size_t limit) {
    MLX_TRY_RETURN_VALUE(0, mlx::core::set_memory_limit(limit));
}

/// Set the MLX allocator cache limit in bytes.
size_t mlx_set_cache_limit(size_t limit) {
    MLX_TRY_RETURN_VALUE(0, mlx::core::set_cache_limit(limit));
}

/// Set the MLX allocator wired limit in bytes.
size_t mlx_set_wired_limit(size_t limit) {
    MLX_TRY_RETURN_VALUE(0, mlx::core::set_wired_limit(limit));
}

/// Release cached Metal buffers and other allocator caches.
/// Equivalent to `mx.metal.clear_cache()` in Python.
void mlx_metal_clear_cache() {
    MLX_TRY_VOID(mlx::core::clear_cache());
}

} // extern "C"
