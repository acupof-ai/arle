use anyhow::Result;

use super::mlx::{
    MlxArray, async_eval, clear_cache, concatenate_axis, matmul, quantized_matmul, zeros,
};
use super::weights::WeightTensor;

#[cfg(feature = "metal")]
pub(super) fn metal_async_eval(arr: &MlxArray) -> Result<()> {
    async_eval(&[arr]);
    Ok(())
}

#[cfg(feature = "metal")]
pub(super) fn clear_metal_cache() {
    clear_cache();
}

#[cfg(feature = "metal")]
pub(super) fn extend_kv_cache(
    cache: &mut MlxArray,
    n_kv_heads: i32,
    head_dim: i32,
    new_cap: i32,
) -> Result<()> {
    let current_cap = cache.shape().get(2).copied().unwrap_or_default();
    if new_cap <= current_cap {
        return Ok(());
    }

    let extra = zeros(
        &[1, n_kv_heads, new_cap - current_cap, head_dim],
        cache.dtype(),
    );
    *cache = concatenate_axis(&[cache.clone(), extra], 2);
    Ok(())
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
    }
}
