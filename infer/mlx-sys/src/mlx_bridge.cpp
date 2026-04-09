#include "mlx_common.h"

extern "C" {

// === Array lifecycle ===

mlx_array* mlx_array_new_float32(float val) {
    return from_arr(array(val));
}

mlx_array* mlx_array_new_int32(int32_t val) {
    return from_arr(array(val));
}

mlx_array* mlx_array_from_data(const void* data, const int32_t* shape, int32_t ndim, int32_t dtype_val) {
    auto sh = make_shape(shape, static_cast<size_t>(ndim));
    auto dt = to_dtype(dtype_val);
    // MLX array constructor needs the allocator to copy data
    size_t nbytes = 1;
    for (int i = 0; i < ndim; i++) nbytes *= shape[i];
    nbytes *= size_of(dt);
    auto buf = allocator::malloc(nbytes);
    std::memcpy(buf.raw_ptr(), data, nbytes);
    return reinterpret_cast<mlx_array*>(new array(std::move(buf), sh, dt));
}

mlx_array* mlx_array_clone(mlx_array* a) {
    // Copy the shared_ptr (increment refcount, same underlying data)
    return reinterpret_cast<mlx_array*>(new array(*to_arr(a)));
}

void mlx_array_free(mlx_array* a) {
    delete to_arr(a);
}

int32_t mlx_array_ndim(mlx_array* a) {
    return static_cast<int32_t>(to_arr(a)->ndim());
}

const int32_t* mlx_array_shape(mlx_array* a) {
    // MLX shape() returns std::vector<int>; data() gives stable pointer
    // while the array is alive.
    return to_arr(a)->shape().data();
}

int32_t mlx_array_dtype(mlx_array* a) {
    return from_dtype(to_arr(a)->dtype());
}

int32_t mlx_array_item_int32(mlx_array* a) {
    return to_arr(a)->item<int32_t>();
}

float mlx_array_item_float32(mlx_array* a) {
    return to_arr(a)->item<float>();
}

const float* mlx_array_data_float32(mlx_array* a) {
    return to_arr(a)->data<float>();
}

size_t mlx_array_size(mlx_array* a) {
    return to_arr(a)->size();
}

// === Binary ops ===

mlx_array* mlx_add(mlx_array* a, mlx_array* b) {
    return from_arr(add(*to_arr(a), *to_arr(b)));
}

mlx_array* mlx_subtract(mlx_array* a, mlx_array* b) {
    return from_arr(subtract(*to_arr(a), *to_arr(b)));
}

mlx_array* mlx_multiply(mlx_array* a, mlx_array* b) {
    return from_arr(multiply(*to_arr(a), *to_arr(b)));
}

mlx_array* mlx_matmul(mlx_array* a, mlx_array* b) {
    return from_arr(matmul(*to_arr(a), *to_arr(b)));
}

mlx_array* mlx_greater(mlx_array* a, mlx_array* b) {
    return from_arr(greater(*to_arr(a), *to_arr(b)));
}

// === Unary ops ===

mlx_array* mlx_exp(mlx_array* a) {
    return from_arr(exp(*to_arr(a)));
}

mlx_array* mlx_log1p(mlx_array* a) {
    return from_arr(log1p(*to_arr(a)));
}

mlx_array* mlx_negative(mlx_array* a) {
    return from_arr(negative(*to_arr(a)));
}

mlx_array* mlx_sqrt(mlx_array* a) {
    return from_arr(sqrt(*to_arr(a)));
}

mlx_array* mlx_reciprocal(mlx_array* a) {
    return from_arr(reciprocal(*to_arr(a)));
}

mlx_array* mlx_sigmoid(mlx_array* a) {
    return from_arr(sigmoid(*to_arr(a)));
}

// === Shape ops ===

mlx_array* mlx_reshape(mlx_array* a, const int32_t* shape, size_t ndim) {
    auto sh = make_shape(shape, ndim);
    return from_arr(reshape(*to_arr(a), sh));
}

mlx_array* mlx_transpose(mlx_array* a) {
    return from_arr(transpose(*to_arr(a)));
}

mlx_array* mlx_transpose_axes(mlx_array* a, const int32_t* axes, size_t n) {
    std::vector<int> ax(axes, axes + n);
    return from_arr(transpose(*to_arr(a), ax));
}

mlx_array* mlx_astype(mlx_array* a, int32_t dtype) {
    return from_arr(astype(*to_arr(a), to_dtype(dtype)));
}

mlx_array* mlx_broadcast_to(mlx_array* a, const int32_t* shape, size_t ndim) {
    auto sh = make_shape(shape, ndim);
    return from_arr(broadcast_to(*to_arr(a), sh));
}

mlx_array* mlx_expand_dims(mlx_array* a, int32_t axis) {
    return from_arr(expand_dims(*to_arr(a), static_cast<int>(axis)));
}

mlx_array* mlx_zeros(const int32_t* shape, size_t ndim, int32_t dtype) {
    auto sh = make_shape(shape, ndim);
    return from_arr(zeros(sh, to_dtype(dtype)));
}

// === Indexing ===

mlx_array* mlx_take_axis(mlx_array* a, mlx_array* indices, int32_t axis) {
    return from_arr(take(*to_arr(a), *to_arr(indices), static_cast<int>(axis)));
}

mlx_array* mlx_slice(mlx_array* a, const int32_t* start, const int32_t* stop,
                     const int32_t* strides, size_t ndim) {
    Shape st; for(size_t i=0;i<ndim;i++) st.push_back(start[i]);
    Shape sp; for(size_t i=0;i<ndim;i++) sp.push_back(stop[i]);
    Shape sr; for(size_t i=0;i<ndim;i++) sr.push_back(strides[i]);
    return from_arr(slice(*to_arr(a), st, sp, sr));
}

mlx_array* mlx_slice_update(mlx_array* src, mlx_array* update,
                            const int32_t* start, const int32_t* stop,
                            const int32_t* strides, size_t ndim) {
    Shape st; for(size_t i=0;i<ndim;i++) st.push_back(start[i]);
    Shape sp; for(size_t i=0;i<ndim;i++) sp.push_back(stop[i]);
    Shape sr; for(size_t i=0;i<ndim;i++) sr.push_back(strides[i]);
    return from_arr(slice_update(*to_arr(src), *to_arr(update), st, sp, sr));
}

mlx_array* mlx_concatenate_axis(mlx_array** arrays, size_t count, int32_t axis) {
    std::vector<array> arrs;
    arrs.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        arrs.push_back(*to_arr(arrays[i]));
    }
    return from_arr(concatenate(arrs, static_cast<int>(axis)));
}

mlx_array* mlx_where(mlx_array* cond, mlx_array* a, mlx_array* b) {
    return from_arr(where(*to_arr(cond), *to_arr(a), *to_arr(b)));
}

// === Reduction ===

mlx_array* mlx_sum_axis(mlx_array* a, int32_t axis, bool keepdims) {
    return from_arr(sum(*to_arr(a), static_cast<int>(axis), keepdims));
}

mlx_array* mlx_argmax(mlx_array* a, bool keepdims) {
    return from_arr(argmax(*to_arr(a), keepdims));
}

// === Quantized ===

mlx_array* mlx_quantized_matmul(mlx_array* x, mlx_array* w, mlx_array* scales,
                                mlx_array* biases, bool transpose,
                                int32_t group_size, int32_t bits) {
    return from_arr(quantized_matmul(
        *to_arr(x), *to_arr(w), *to_arr(scales), *to_arr(biases),
        transpose, group_size, bits));
}

mlx_array* mlx_dequantize(mlx_array* w, mlx_array* scales, mlx_array* biases,
                          int32_t group_size, int32_t bits) {
    return from_arr(dequantize(*to_arr(w), *to_arr(scales), *to_arr(biases),
                               group_size, bits));
}

// === Fast ops ===

mlx_array* mlx_fast_rms_norm(mlx_array* x, mlx_array* weight, float eps) {
    if (weight == nullptr) {
        return from_arr(fast::rms_norm(*to_arr(x), std::nullopt, eps));
    }
    return from_arr(fast::rms_norm(*to_arr(x), *to_arr(weight), eps));
}

mlx_array* mlx_fast_rope(mlx_array* x, int32_t dims, bool traditional,
                         float base, float scale, int32_t offset) {
    return from_arr(fast::rope(
        *to_arr(x), dims, traditional, base, scale, offset));
}

mlx_array* mlx_fast_sdpa(mlx_array* q, mlx_array* k, mlx_array* v,
                         float scale, const char* mask_mode) {
    // mask_mode: "" for no mask, "causal" for causal masking.
    // NEVER pass nullptr — std::string(nullptr) is UB.
    std::string mode(mask_mode);
    return from_arr(fast::scaled_dot_product_attention(
        *to_arr(q), *to_arr(k), *to_arr(v), scale, mode));
}

// === Random ===

mlx_array* mlx_random_categorical(mlx_array* logits, int32_t axis) {
    return from_arr(random::categorical(*to_arr(logits), static_cast<int>(axis)));
}

// === Transforms ===

void mlx_eval(mlx_array** arrays, size_t count) {
    std::vector<array> arrs;
    arrs.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        arrs.push_back(*to_arr(arrays[i]));
    }
    eval(arrs);
}

void mlx_async_eval(mlx_array** arrays, size_t count) {
    std::vector<array> arrs;
    arrs.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        arrs.push_back(*to_arr(arrays[i]));
    }
    async_eval(arrs);
}

// === IO ===

int32_t mlx_load_safetensors(const char* path,
                             const char*** out_names,
                             mlx_array*** out_arrays) {
    auto result = load_safetensors(std::string(path));
    // result is SafetensorsLoad: { .data = unordered_map<string,array>, .metadata = ... }
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

// === Conv1d ===

mlx_array* mlx_conv1d(mlx_array* input, mlx_array* weight,
                      int32_t stride, int32_t padding,
                      int32_t dilation, int32_t groups) {
    return from_arr(conv1d(*to_arr(input), *to_arr(weight),
                           stride, padding, dilation, groups));
}

// === Metal kernel ===

// Opaque handle wrapping the fast::MetalKernel builder
struct mlx_metal_kernel {
    std::string name;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string source;
    std::string header;
};

void* mlx_metal_kernel_new(const char* name,
                           const char** input_names, size_t n_inputs,
                           const char** output_names, size_t n_outputs,
                           const char* source, const char* header) {
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
}

void mlx_metal_kernel_free(void* kernel) {
    delete static_cast<mlx_metal_kernel*>(kernel);
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
    auto* k = static_cast<mlx_metal_kernel*>(kernel);

    // Build input array vector
    std::vector<array> in_arrs;
    in_arrs.reserve(n_inputs);
    for (size_t i = 0; i < n_inputs; ++i) {
        in_arrs.push_back(*to_arr(inputs[i]));
    }

    // Build output shapes (using Shape = SmallVector<int>)
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

    // Build template args as vector of (name, TemplateArg) pairs
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

    // Create the kernel function
    auto mk = fast::metal_kernel(
        k->name,
        k->input_names,
        k->output_names,
        k->source,
        k->header,
        true,   // ensure_row_contiguous
        false   // atomic_outputs
    );

    // Set grid and threadgroup
    auto grid_tuple = std::make_tuple(grid[0], grid[1], grid[2]);
    auto tg_tuple = std::make_tuple(threadgroup[0], threadgroup[1], threadgroup[2]);

    // Apply — CustomKernelFunction signature:
    // (inputs, output_shapes, output_dtypes, grid, threadgroup,
    //  template_args, init_value, atomic_outputs, stream)
    auto result = mk(
        in_arrs, out_shapes, out_dtypes,
        grid_tuple, tg_tuple,
        template_args,
        std::nullopt,  // init_value
        false,         // atomic_outputs
        {}             // default stream
    );

    // Write results to output array
    for (size_t i = 0; i < result.size() && i < n_outputs; ++i) {
        outputs[i] = from_arr(std::move(result[i]));
    }
}

} // extern "C"
