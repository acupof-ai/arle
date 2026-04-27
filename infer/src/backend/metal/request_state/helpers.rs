//! Top-level helper functions for `request_state.rs`.
//!
//! Split out of `request_state.rs` (pure structural refactor — no behavior change).
//! Pure functions for KV padding/stripping + the qwen35 trace toggle.

use super::super::KV_CACHE_CHUNK;
use super::super::mlx::{MlxArray, slice, zeros};

pub(super) fn metal_qwen35_trace_enabled() -> bool {
    std::env::var("AGENT_INFER_METAL_QWEN35_TRACE")
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
}

pub(super) fn round_up_kv_capacity(tokens: i32) -> i32 {
    ((tokens + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK) * KV_CACHE_CHUNK
}

pub(super) fn left_pad_kv_cache_row(
    array: &MlxArray,
    left_pad: i32,
    cache_len: i32,
    target_kv_capacity: i32,
) -> MlxArray {
    let shape = array.shape();
    debug_assert_eq!(shape.len(), 4);
    debug_assert_eq!(shape[0], 1);
    debug_assert!(left_pad >= 0);
    debug_assert!(cache_len >= 0);
    debug_assert!(left_pad + cache_len <= target_kv_capacity);

    let n_kv = shape[1];
    let head_dim = shape[3];
    let mut padded = zeros(&[1, n_kv, target_kv_capacity, head_dim], array.dtype());
    if cache_len == 0 {
        return padded;
    }

    let valid = slice(
        array,
        &[0, 0, 0, 0],
        &[1, n_kv, cache_len, head_dim],
        &[1, 1, 1, 1],
    );
    padded = super::super::mlx::slice_update(
        &mut padded,
        &valid,
        &[0, 0, left_pad, 0],
        &[1, n_kv, left_pad + cache_len, head_dim],
    );
    padded
}

pub(super) fn strip_left_padding_from_packed_row(
    array: &MlxArray,
    row: i32,
    left_pad: i32,
    batch_cache_len: i32,
    row_kv_capacity: i32,
) -> MlxArray {
    let row_slice = slice_row(array, row);
    let shape = row_slice.shape();
    debug_assert_eq!(shape.len(), 4);
    debug_assert_eq!(shape[0], 1);
    debug_assert!(left_pad >= 0);
    debug_assert!(batch_cache_len >= left_pad);

    let n_kv = shape[1];
    let head_dim = shape[3];
    let valid_len = batch_cache_len - left_pad;
    let mut unpadded = zeros(&[1, n_kv, row_kv_capacity, head_dim], row_slice.dtype());
    if valid_len == 0 {
        return unpadded;
    }

    let valid = slice(
        &row_slice,
        &[0, 0, left_pad, 0],
        &[1, n_kv, batch_cache_len, head_dim],
        &[1, 1, 1, 1],
    );
    unpadded = super::super::mlx::slice_update(
        &mut unpadded,
        &valid,
        &[0, 0, 0, 0],
        &[1, n_kv, valid_len, head_dim],
    );
    unpadded
}

pub(super) fn slice_row(array: &MlxArray, row: i32) -> MlxArray {
    let mut start = vec![0; array.shape().len()];
    let mut end = array.shape().to_vec();
    let strides = vec![1; array.shape().len()];
    start[0] = row;
    end[0] = row + 1;
    slice(array, &start, &end, &strides)
}
