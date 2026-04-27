use super::mlx::{
    MlxArray, async_eval, clear_cache, concatenate_axis, gguf_quantized_matmul, matmul,
    quantized_matmul, reshape, transpose_axes, zeros,
};
use super::weights::WeightTensor;

#[cfg(feature = "metal")]
pub(super) fn metal_async_eval(arr: &MlxArray) {
    async_eval(&[arr]);
}

#[cfg(feature = "metal")]
pub(super) fn clear_metal_cache() {
    clear_cache();
}

#[cfg(feature = "metal")]
pub(super) fn extend_kv_cache(cache: &mut MlxArray, n_kv_heads: i32, head_dim: i32, new_cap: i32) {
    let current_cap = cache.shape().get(2).copied().unwrap_or_default();
    if new_cap <= current_cap {
        return;
    }

    // Inherit the batch dim from the existing cache — in the packed-decode
    // path this cache holds multiple rows stacked along axis 0.
    let batch = cache.shape().first().copied().unwrap_or(1);
    let extra = zeros(
        &[batch, n_kv_heads, new_cap - current_cap, head_dim],
        cache.dtype(),
    );
    *cache = concatenate_axis(&[cache.clone(), extra], 2);
}

/// `x @ weight.T` — no bias, dispatches to dense matmul or quantized matmul.
///
/// For `Dense(w_t)`, `w_t` is already transposed at load time (shape `[in, out]`),
/// so this is just `matmul(x, w_t)` without an extra transpose.
#[cfg(feature = "metal")]
#[inline]
pub(super) fn linear(x: &MlxArray, weight: &WeightTensor) -> MlxArray {
    match weight {
        WeightTensor::Dense(w_t) => {
            // w_t is pre-transposed [in, out]; direct matmul, no per-call transpose.
            matmul(x, w_t)
        }
        WeightTensor::Quantized {
            w,
            scales,
            biases,
            group_size,
            bits,
        } => {
            // w stored as [out, in] packed uint32; transpose=true → x @ w.T
            quantized_matmul(x, w, scales, biases, true, *group_size, *bits)
        }
        WeightTensor::GgufPacked {
            w,
            format,
            rows,
            cols,
        } => gguf_quantized_matmul(x, w, format.as_i32(), *rows, *cols),
        WeightTensor::GgufPackedInputReordered {
            w,
            format,
            rows,
            cols,
            num_key_heads,
            num_value_heads_per_key,
            head_dim,
        } => {
            let x_reordered =
                reorder_qwen35_v_cols_input(x, *num_key_heads, *num_value_heads_per_key, *head_dim);
            gguf_quantized_matmul(&x_reordered, w, format.as_i32(), *rows, *cols)
        }
    }
}

#[cfg(feature = "metal")]
fn reorder_qwen35_v_cols_input(
    x: &MlxArray,
    num_key_heads: i32,
    num_value_heads_per_key: i32,
    head_dim: i32,
) -> MlxArray {
    if num_value_heads_per_key <= 1 {
        return x.clone();
    }

    let shape = x.shape();
    let Some(&cols) = shape.last() else {
        return x.clone();
    };
    assert_eq!(
        cols,
        num_key_heads * num_value_heads_per_key * head_dim,
        "Qwen3.5 GGUF value-head input reorder dimension mismatch"
    );

    let prefix_ndim = shape.len() - 1;
    let mut expanded = shape[..prefix_ndim].to_vec();
    expanded.extend([num_key_heads, num_value_heads_per_key, head_dim]);

    let mut axes: Vec<i32> = (0..i32::try_from(prefix_ndim).expect("ndim fits i32")).collect();
    let base = i32::try_from(prefix_ndim).expect("ndim fits i32");
    axes.extend([base + 1, base, base + 2]);

    let expanded_x = reshape(x, &expanded);
    reshape(&transpose_axes(&expanded_x, &axes), shape)
}
