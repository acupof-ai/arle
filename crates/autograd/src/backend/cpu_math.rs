//! CPU math helper functions extracted from `backend.rs`.
//!
//! All functions here are pure host-side math; they have no dependency on the
//! `Backend` trait or any device handle type.  The module is included via
//! `#[path = "backend/cpu_math.rs"] mod cpu_math;` at the end of `backend.rs`
//! and its public symbols re-exported into the `backend` module namespace.

use crate::{AutogradError, Result};

// Type aliases shared between cpu_qwen_decode_prepare_q/kv (not re-exported;
// callers above us see the concrete tuple).
type QwenDecodePrepareQHost = (Vec<f32>, Option<Vec<f32>>, Vec<usize>);
type QwenDecodePrepareKvHost = (Vec<f32>, Vec<f32>, Vec<usize>);

// Re-export the CausalSdpaHostGradTriplet from parent so we don't have to
// qualify it here.
use super::CausalSdpaHostGradTriplet;

pub(crate) fn shape_size(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Truncate f32 to bf16 by keeping the upper 16 bits. Exact for values that
/// were originally bf16 (lower 16 bits are zero).
pub fn f32_to_bf16_bits(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

pub fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = i32::from((bits >> 3) & 0x0f);
    let mant = i32::from(bits & 0x07);
    if exp == 0 {
        if mant == 0 {
            return sign * 0.0;
        }
        return sign * (mant as f32 / 8.0) * 2.0_f32.powi(-6);
    }
    if exp == 0x0f && mant == 0x07 {
        return f32::NAN;
    }
    sign * (1.0 + mant as f32 / 8.0) * 2.0_f32.powi(exp - 7)
}

pub fn dequantize_fp8_block_scaled_host(
    weight: &[u8],
    scales: &[f32],
    shape: &[usize],
    block_m: usize,
    block_k: usize,
) -> Result<Vec<f32>> {
    validate_fp8_block_scaled(weight, scales, shape, block_m, block_k)?;
    let rows = shape[0];
    let cols = shape[1];
    let scale_cols = cols.div_ceil(block_k);
    let out = (0..rows)
        .flat_map(|row| {
            (0..cols).map(move |col| {
                let scale = scales[(row / block_m) * scale_cols + (col / block_k)];
                fp8_e4m3_to_f32(weight[row * cols + col]) * scale
            })
        })
        .collect();
    Ok(out)
}

pub fn validate_fp8_block_scaled(
    weight: &[u8],
    scales: &[f32],
    shape: &[usize],
    block_m: usize,
    block_k: usize,
) -> Result<()> {
    if shape.len() != 2 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "2",
            got: shape.len(),
        });
    }
    if block_m == 0 || block_k == 0 {
        return Err(crate::AutogradError::TapeInvariant(
            "fp8 block-scaled block_m/block_k must be non-zero",
        ));
    }
    let expected_weight = shape_size(shape);
    if weight.len() != expected_weight {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: weight.len(),
            shape: shape.to_vec(),
            size: expected_weight,
        });
    }
    let scale_shape = vec![shape[0].div_ceil(block_m), shape[1].div_ceil(block_k)];
    let expected_scales = shape_size(&scale_shape);
    if scales.len() != expected_scales {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: scales.len(),
            shape: scale_shape,
            size: expected_scales,
        });
    }
    Ok(())
}

/// CPU reference implementation of row-major matmul (2D + batched 3D).
/// Exposed so other backends can reuse it as a fallback.
pub fn cpu_matmul_forward(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>)> {
    use crate::AutogradError;
    match (a_shape.len(), b_shape.len()) {
        (2, 2) => {
            let out_shape = matmul_output_shape(a_shape, b_shape)?;
            let m = a_shape[0];
            let k = a_shape[1];
            let n = b_shape[1];
            let mut out = vec![0.0f32; m * n];
            sgemm_row_major(m, k, n, a, b, &mut out);
            Ok((out, out_shape))
        }
        (3, 3) => {
            let out_shape = matmul_output_shape(a_shape, b_shape)?;
            let batch = a_shape[0];
            let m = a_shape[1];
            let k = a_shape[2];
            let n = b_shape[2];
            let mut out = vec![0.0f32; batch * m * n];
            let a_batch_stride = m * k;
            let b_batch_stride = k * n;
            let out_batch_stride = m * n;
            for batch_index in 0..batch {
                let a_base = batch_index * a_batch_stride;
                let b_base = batch_index * b_batch_stride;
                let out_base = batch_index * out_batch_stride;
                sgemm_row_major(
                    m,
                    k,
                    n,
                    &a[a_base..a_base + a_batch_stride],
                    &b[b_base..b_base + b_batch_stride],
                    &mut out[out_base..out_base + out_batch_stride],
                );
            }
            Ok((out, out_shape))
        }
        _ => Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2 or rank-3",
            got: a_shape.len().max(b_shape.len()),
        }),
    }
}

pub fn matmul_bt_output_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    use crate::AutogradError;
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2",
            got: a_shape.len().max(b_shape.len()),
        });
    }
    let k_a = a_shape[1];
    let k_b = b_shape[1];
    if k_a != k_b {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![k_a],
            got: vec![k_b],
        });
    }
    Ok(vec![a_shape[0], b_shape[0]])
}

/// CPU `C = A @ B^T` for rank-2 row-major tensors without materialising `B^T`.
/// Shapes: `A:[M,K]`, `B:[N,K]`, output `[M,N]`.
pub fn cpu_matmul_bt_forward(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>)> {
    let out_shape = matmul_bt_output_shape(a_shape, b_shape)?;
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];
    let expected_a = m * k;
    let expected_b = n * k;
    if a.len() != expected_a {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: a.len(),
            shape: a_shape.to_vec(),
            size: expected_a,
        });
    }
    if b.len() != expected_b {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: b.len(),
            shape: b_shape.to_vec(),
            size: expected_b,
        });
    }
    let mut out = vec![0.0f32; m * n];
    matmul_a_bt_into(a, a_shape_2d(m, k), b, b_shape_2d(n, k), &mut out);
    Ok((out, out_shape))
}

/// Row-major `C = A @ B` for one rank-2 sgemm tile. OPD-shape-aware dispatch:
///   - **Saxpy inline loop** for thin matmuls (`n < SAXPY_N_THRESHOLD`) and
///     single-row matmuls (`m == 1`). Hits ~20 GFLOPs/s on Zen 2 for
///     cache-resident OPD projection shapes, and avoids matrixmultiply's
///     pack overhead in the M=1 rollout-last-row lm_head regime.
///   - **`matrixmultiply::sgemm`** for `n >= SAXPY_N_THRESHOLD`. With
///     `lm_head`'s `N=151936` the saxpy thrashes L1 (608 KB per B row);
///     matrixmultiply's tile-pack reuses A across N-tiles and pushes lm_head
///     forward from ~8 GFLOPs/s saxpy ceiling to ~16 GFLOPs/s on Zen 2.
///
/// Caller guarantees `a.len() == m*k`, `b.len() == k*n`, `out.len() == m*n` and
/// that `out` is already zero-initialised.
fn sgemm_row_major(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], out: &mut [f32]) {
    /// Crossover N where matrixmultiply's pack-A / pack-B is amortised by
    /// the extra cache reuse. Empirically at M=4 Qwen3-0.6B shapes on Zen 2:
    /// gate_proj (N=3072) wins with saxpy; lm_head (N=151936) wins with
    /// matrixmultiply. Loose bracket so future model shapes between 3K and
    /// 30K take the lower-overhead saxpy path.
    const SAXPY_N_THRESHOLD: usize = 32_768;
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    if m == 1 || n < SAXPY_N_THRESHOLD {
        for row in 0..m {
            let a_row = &a[row * k..(row + 1) * k];
            let out_row = &mut out[row * n..(row + 1) * n];
            for inner in 0..k {
                let a_value = a_row[inner];
                let b_row = &b[inner * n..(inner + 1) * n];
                for col in 0..n {
                    out_row[col] += a_value * b_row[col];
                }
            }
        }
        return;
    }
    // Safety: a/b are row-major contiguous slices of length m*k / k*n; `out`
    // is row-major contiguous m*n; beta=0 means the pre-existing `C` values
    // are unread.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// CPU reference matmul backward. Computes `grad_a = grad_out @ B^T` and
/// `grad_b = A^T @ grad_out`. Physically transposes the last two axes of the
/// saved operand on the host and then calls `cpu_matmul_forward` — this is
/// the authoritative numerical reference every GPU backend must match.
///
/// `need_grad_a`/`need_grad_b` skip the corresponding SGEMM when false; the
/// returned `Vec<f32>` is empty in that case so callers can cheaply detect
/// "no grad produced" without allocating.
pub fn cpu_matmul_backward(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    grad_out: &[f32],
    grad_out_shape: &[usize],
    need_grad_a: bool,
    need_grad_b: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    use crate::AutogradError;
    let expected_out = matmul_output_shape(a_shape, b_shape)?;
    if grad_out_shape != expected_out.as_slice() {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_out,
            got: grad_out_shape.to_vec(),
        });
    }

    match (a_shape.len(), b_shape.len()) {
        (2, 2) => {
            let m = a_shape[0];
            let k = a_shape[1];
            let n = b_shape[1];
            let grad_a = if need_grad_a {
                let mut out = vec![0.0f32; m * k];
                matmul_a_bt_into(grad_out, a_shape_2d(m, n), b, b_shape_2d(k, n), &mut out);
                out
            } else {
                Vec::new()
            };
            let grad_b = if need_grad_b {
                let mut out = vec![0.0f32; k * n];
                matmul_at_b_into(a, a_shape_2d(m, k), grad_out, b_shape_2d(m, n), &mut out);
                out
            } else {
                Vec::new()
            };
            Ok((grad_a, grad_b))
        }
        (3, 3) => {
            let batch = a_shape[0];
            let m = a_shape[1];
            let k = a_shape[2];
            let n = b_shape[2];
            let a_plane = m * k;
            let b_plane = k * n;
            let grad_out_plane = m * n;
            let grad_a = if need_grad_a {
                let mut out = vec![0.0f32; batch * m * k];
                for bi in 0..batch {
                    matmul_a_bt_into(
                        &grad_out[bi * grad_out_plane..(bi + 1) * grad_out_plane],
                        a_shape_2d(m, n),
                        &b[bi * b_plane..(bi + 1) * b_plane],
                        b_shape_2d(k, n),
                        &mut out[bi * a_plane..(bi + 1) * a_plane],
                    );
                }
                out
            } else {
                Vec::new()
            };
            let grad_b = if need_grad_b {
                let mut out = vec![0.0f32; batch * k * n];
                for bi in 0..batch {
                    matmul_at_b_into(
                        &a[bi * a_plane..(bi + 1) * a_plane],
                        a_shape_2d(m, k),
                        &grad_out[bi * grad_out_plane..(bi + 1) * grad_out_plane],
                        b_shape_2d(m, n),
                        &mut out[bi * b_plane..(bi + 1) * b_plane],
                    );
                }
                out
            } else {
                Vec::new()
            };
            Ok((grad_a, grad_b))
        }
        _ => Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2 or rank-3",
            got: a_shape.len().max(b_shape.len()),
        }),
    }
}

pub fn cpu_matmul_bt_backward(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    grad_out: &[f32],
    grad_out_shape: &[usize],
    need_grad_a: bool,
    need_grad_b: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    use crate::AutogradError;
    let expected_out = matmul_bt_output_shape(a_shape, b_shape)?;
    if grad_out_shape != expected_out.as_slice() {
        return Err(AutogradError::ShapeMismatch {
            expected: expected_out,
            got: grad_out_shape.to_vec(),
        });
    }

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];
    let grad_a = if need_grad_a {
        let (grad_a, _) = cpu_matmul_forward(grad_out, &[m, n], b, &[n, k])?;
        grad_a
    } else {
        Vec::new()
    };
    let grad_b = if need_grad_b {
        let mut out = vec![0.0f32; n * k];
        matmul_at_b_into(grad_out, a_shape_2d(m, n), a, b_shape_2d(m, k), &mut out);
        out
    } else {
        Vec::new()
    };
    Ok((grad_a, grad_b))
}

/// Pair of rank-2 row dimensions, kept inline so the rank-3 dispatcher can
/// reuse the same kernel without re-allocating shape `Vec`s on every batch.
#[inline]
fn a_shape_2d(rows: usize, cols: usize) -> (usize, usize) {
    (rows, cols)
}
#[inline]
fn b_shape_2d(rows: usize, cols: usize) -> (usize, usize) {
    (rows, cols)
}

/// Compute `out = a @ b^T` for row-major rank-2 buffers **without materialising
/// `b^T`**. Dispatches to `matrixmultiply::sgemm` with a strided view of `b`:
/// passing `rsb = 1, csb = N_phys` re-interprets the row-major `[K_phys, N_phys]`
/// buffer as the transposed `[N_phys, K_phys]` matrix without copying. Used by
/// `cpu_matmul_backward` for `grad_a = grad_out @ B^T`.
///
/// Shapes (caller-enforced):
/// - `a`: `[M, N]` (row-major contiguous, len `M * N`)
/// - `b`: `[K, N]` (row-major contiguous, len `K * N`) — logical pre-transpose
/// - `out`: `[M, K]` (row-major contiguous, len `M * K`, pre-zeroed)
#[inline]
fn matmul_a_bt_into(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
    out: &mut [f32],
) {
    let (m, n_a) = a_shape;
    let (k, n_b) = b_shape;
    debug_assert_eq!(n_a, n_b, "a and b must share the K-equivalent dim");
    let n = n_a;
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    // Safety: a/b are row-major contiguous slices of length m*n / k*n; the
    // strided `b` view (rsb=1, csb=n) addresses every element once via
    // `b_ptr[k_log * 1 + n_log * n]`, which equals `b_ptr[n_log * n + k_log]`
    // = the (n_log, k_log) entry of the logical transpose. beta=0 means the
    // pre-existing `out` values are unread.
    unsafe {
        matrixmultiply::sgemm(
            m,
            n,
            k,
            1.0,
            a.as_ptr(),
            n as isize,
            1,
            b.as_ptr(),
            1,
            n as isize,
            0.0,
            out.as_mut_ptr(),
            k as isize,
            1,
        );
    }
}

/// Compute `out = a^T @ b` for row-major rank-2 buffers **without materialising
/// `a^T`**. Dispatches to `matrixmultiply::sgemm` with a strided view of `a`:
/// passing `rsa = 1, csa = K_phys` re-interprets the row-major `[M_phys, K_phys]`
/// buffer as the transposed `[K_phys, M_phys]` matrix without copying. Used by
/// `cpu_matmul_backward` for `grad_b = A^T @ grad_out`.
///
/// Shapes (caller-enforced):
/// - `a`: `[M, K]` (row-major contiguous, len `M * K`) — logical pre-transpose
/// - `b`: `[M, N]` (row-major contiguous, len `M * N`)
/// - `out`: `[K, N]` (row-major contiguous, len `K * N`, pre-zeroed)
#[inline]
fn matmul_at_b_into(
    a: &[f32],
    a_shape: (usize, usize),
    b: &[f32],
    b_shape: (usize, usize),
    out: &mut [f32],
) {
    let (m_a, k) = a_shape;
    let (m_b, n) = b_shape;
    debug_assert_eq!(m_a, m_b, "a and b must share the M dim");
    let m = m_a;
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    // Safety: a/b are row-major contiguous slices of length m*k / m*n; the
    // strided `a` view (rsa=1, csa=k) addresses `a_ptr[k_log * 1 + m_log * k]`
    // = `a_ptr[m_log * k + k_log]` = the (m_log, k_log) entry of physical A,
    // which is the (k_log, m_log) entry of logical A^T. beta=0 means the
    // pre-existing `out` values are unread.
    unsafe {
        matrixmultiply::sgemm(
            k,
            m,
            n,
            1.0,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// CPU reference for row-wise softmax over the last axis. Matches the
/// numerically-stable implementation in `ops::softmax::softmax` so that
/// backends can fall back to this when GPU acceleration is unavailable.
pub fn cpu_softmax_forward_last_axis(x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let rows = x.len() / last_dim;
    let mut out = vec![0.0f32; x.len()];
    for row in 0..rows {
        let base = row * last_dim;
        let slice = &x[base..base + last_dim];
        let max_value = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = slice
            .iter()
            .map(|value| (*value - max_value).exp())
            .sum::<f32>();
        for col in 0..last_dim {
            out[base + col] = (slice[col] - max_value).exp() / denom;
        }
    }
    Ok(out)
}

/// CPU reference for row-wise log-softmax over the last axis.
pub fn cpu_log_softmax_forward_last_axis(x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let rows = x.len() / last_dim;
    let mut out = vec![0.0f32; x.len()];
    for row in 0..rows {
        let base = row * last_dim;
        let slice = &x[base..base + last_dim];
        let max_value = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = slice
            .iter()
            .map(|value| (*value - max_value).exp())
            .sum::<f32>();
        let log_denom = denom.ln();
        for col in 0..last_dim {
            out[base + col] = (slice[col] - max_value) - log_denom;
        }
    }
    Ok(out)
}

/// CPU reference for `log_softmax_last_axis_backward`. Computes
/// `grad_input[i, j] = upstream[i, j] - exp(log_softmax_output[i, j]) * sum_j(upstream[i, j])`
/// row-wise over the last axis. `log_softmax_output` is the saved
/// forward output — `softmax(x) = exp(log_softmax(x))`, so the
/// derivative identity reuses it without recomputing softmax.
///
/// Mirrors the inline math in `ops::softmax::log_softmax_backward`
/// (host-eager path). Kept as a free function so the device-handle
/// fallback in `Backend::log_softmax_last_axis_backward` can reuse the
/// same reference and parity tests can compare device against CPU.
pub fn cpu_log_softmax_backward(
    upstream: &[f32],
    log_softmax_output: &[f32],
    shape: &[usize],
) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected = shape_size(shape);
    if upstream.len() != expected {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: upstream.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }
    if log_softmax_output.len() != expected {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: log_softmax_output.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }
    let rows = expected / last_dim;
    let mut grad = vec![0.0_f32; expected];
    for row in 0..rows {
        let base = row * last_dim;
        let mut sum_grad = 0.0_f32;
        for col in 0..last_dim {
            sum_grad += upstream[base + col];
        }
        for col in 0..last_dim {
            grad[base + col] =
                upstream[base + col] - log_softmax_output[base + col].exp() * sum_grad;
        }
    }
    Ok(grad)
}

/// CPU reference for `softmax_last_axis_backward`. Computes
/// `grad_input[i, j] = y[i, j] * (upstream[i, j] - sum_j(upstream[i, j] * y[i, j]))`
/// row-wise over the last axis. `softmax_output` is the saved forward output.
pub fn cpu_softmax_backward(
    upstream: &[f32],
    softmax_output: &[f32],
    shape: &[usize],
) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected = shape_size(shape);
    if upstream.len() != expected {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: upstream.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }
    if softmax_output.len() != expected {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: softmax_output.len(),
            shape: shape.to_vec(),
            size: expected,
        });
    }
    let rows = expected / last_dim;
    let mut grad = vec![0.0_f32; expected];
    for row in 0..rows {
        let base = row * last_dim;
        let mut dot = 0.0_f32;
        for col in 0..last_dim {
            dot += upstream[base + col] * softmax_output[base + col];
        }
        for col in 0..last_dim {
            grad[base + col] = softmax_output[base + col] * (upstream[base + col] - dot);
        }
    }
    Ok(grad)
}

/// CPU reference `out = a * b` for equal-length contiguous slices.
pub fn cpu_mul_forward(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![a.len()],
            got: vec![b.len()],
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
}

/// CPU reference `out = a * s`.
pub fn cpu_mul_scalar_forward(a: &[f32], s: f32) -> Result<Vec<f32>> {
    Ok(a.iter().map(|x| x * s).collect())
}

/// CPU reference right-aligned broadcast-add.
///
/// Output shape equals `a_shape`; `b` is broadcast into `a`. `b_shape.len()`
/// must be `<= a_shape.len()`; each matching `b`-axis must be either `1` or
/// equal to the corresponding `a`-axis. See `broadcast_offset` for the
/// index rule.
pub fn cpu_add_broadcast_forward(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<Vec<f32>> {
    validate_broadcast(a_shape, b_shape)?;
    let a_size: usize = shape_size(a_shape);
    let b_size: usize = shape_size(b_shape);
    if a.len() != a_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: a.len(),
            shape: a_shape.to_vec(),
            size: a_size,
        });
    }
    if b.len() != b_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: b.len(),
            shape: b_shape.to_vec(),
            size: b_size,
        });
    }
    let mut out = vec![0.0f32; a_size];
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = a[index] + b[broadcast_offset(index, a_shape, b_shape)];
    }
    Ok(out)
}

/// Validate that `b_shape` is right-aligned broadcast-compatible into `a_shape`.
pub(crate) fn validate_broadcast(a_shape: &[usize], b_shape: &[usize]) -> Result<()> {
    if b_shape.len() > a_shape.len() {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: a_shape.to_vec(),
            got: b_shape.to_vec(),
        });
    }

    let rank_offset = a_shape.len() - b_shape.len();
    for (index, &dim) in b_shape.iter().enumerate() {
        let target = a_shape[rank_offset + index];
        if dim != 1 && dim != target {
            return Err(crate::AutogradError::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            });
        }
    }

    Ok(())
}

/// Map an output linear index in `out_shape` to the corresponding flat offset
/// into a right-aligned broadcast operand with shape `b_shape`.
pub(crate) fn broadcast_offset(out_index: usize, out_shape: &[usize], b_shape: &[usize]) -> usize {
    if b_shape.is_empty() {
        return 0;
    }

    let coords = linear_to_coords(out_index, out_shape);
    let rank_offset = out_shape.len() - b_shape.len();
    let b_strides = broadcast_strides(b_shape);
    let mut offset = 0usize;
    for (index, stride) in b_strides.iter().enumerate() {
        let coord = if b_shape[index] == 1 {
            0
        } else {
            coords[rank_offset + index]
        };
        offset += coord * stride;
    }
    offset
}

/// Row-major contiguous strides for `shape`. Shared helper used by broadcast
/// math (not the `Tensor` layout stride — that lives in `tensor.rs`).
pub(crate) fn broadcast_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for (index, dim) in shape.iter().enumerate().rev() {
        strides[index] = stride;
        stride *= *dim;
    }
    strides
}

/// Unravel a linear index into per-axis coordinates (row-major).
pub(crate) fn linear_to_coords(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut coords = vec![0; shape.len()];
    for index in (0..shape.len()).rev() {
        let dim = shape[index];
        coords[index] = linear % dim;
        linear /= dim;
    }
    coords
}

/// CPU reference `out = exp(a)`.
pub fn cpu_exp_forward(a: &[f32]) -> Result<Vec<f32>> {
    Ok(a.iter().map(|x| x.exp()).collect())
}

/// CPU reference `out = -a`.
pub fn cpu_neg_forward(a: &[f32]) -> Result<Vec<f32>> {
    Ok(a.iter().map(|x| -x).collect())
}

/// CPU reference `out = |a|`.
pub fn cpu_abs_forward(a: &[f32]) -> Result<Vec<f32>> {
    Ok(a.iter().map(|x| x.abs()).collect())
}

/// CPU reference `sign(x)` with `sign(0) = 0` — the L1 subgradient `abs`
/// picks. `f32::signum` cannot be used: it maps ±0.0 to ±1.0.
pub fn cpu_sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// CPU reference GELU (tanh approximation). Matches the CUDA `gelu_f32` kernel.
pub fn cpu_gelu_forward(a: &[f32]) -> Result<Vec<f32>> {
    const K: f32 = 0.797_884_6_f32; // sqrt(2/pi)
    Ok(a.iter()
        .map(|&x| {
            let inner = K * (x + 0.044_715_f32 * x * x * x);
            0.5_f32 * x * (1.0_f32 + inner.tanh())
        })
        .collect())
}

/// CPU reference SiLU (Swish): `out = a * sigmoid(a)`.
pub fn cpu_silu_forward(a: &[f32]) -> Result<Vec<f32>> {
    Ok(a.iter()
        .map(|&x| x * (1.0_f32 / (1.0_f32 + (-x).exp())))
        .collect())
}

/// CPU reference sigmoid: `out = 1 / (1 + exp(-a))`.
pub fn cpu_sigmoid_forward(a: &[f32]) -> Result<Vec<f32>> {
    Ok(a.iter()
        .map(|&x| 1.0_f32 / (1.0_f32 + (-x).exp()))
        .collect())
}

/// CPU reference RMSNorm over the last axis.
pub fn cpu_rms_norm_forward(
    x: &[f32],
    weight: &[f32],
    shape: &[usize],
    eps: f32,
) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected: usize = shape.iter().product();
    if x.len() != expected {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![x.len()],
        });
    }
    if weight.len() != last_dim {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![last_dim],
            got: vec![weight.len()],
        });
    }

    let rows = expected / last_dim;
    let mut out = vec![0.0_f32; expected];
    for row in 0..rows {
        let base = row * last_dim;
        let slice = &x[base..base + last_dim];
        let mean_sq = slice.iter().map(|v| v * v).sum::<f32>() / last_dim as f32;
        let inv_rms = (mean_sq + eps).sqrt().recip();
        for col in 0..last_dim {
            out[base + col] = slice[col] * inv_rms * weight[col];
        }
    }
    Ok(out)
}

/// CPU reference embedding gather. Returns `[n_ids * dim]` row-major; ids out
/// of range produce a zero row (matches the CUDA kernel's behavior).
pub fn cpu_embedding_forward(
    weight: &[f32],
    vocab: usize,
    dim: usize,
    ids: &[i32],
) -> Result<Vec<f32>> {
    if weight.len() != vocab * dim {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![vocab * dim],
            got: vec![weight.len()],
        });
    }
    let mut out = vec![0.0_f32; ids.len() * dim];
    for (row, &id) in ids.iter().enumerate() {
        if id < 0 {
            continue;
        }
        let id = id as usize;
        if id >= vocab {
            continue;
        }
        let src = &weight[id * dim..(id + 1) * dim];
        let dst = &mut out[row * dim..(row + 1) * dim];
        dst.copy_from_slice(src);
    }
    Ok(out)
}

/// CPU reference sum over the last axis.
pub fn cpu_sum_last_axis_forward(x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    if last_dim == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-zero last dim",
            got: 0,
        });
    }
    let expected: usize = shape.iter().product();
    if x.len() != expected {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![x.len()],
        });
    }
    let rows = expected / last_dim;
    let mut out = vec![0.0_f32; rows];
    for (row, slot) in out.iter_mut().enumerate().take(rows) {
        let base = row * last_dim;
        *slot = x[base..base + last_dim].iter().sum();
    }
    Ok(out)
}

/// CPU reference mean over the last axis.
pub fn cpu_mean_last_axis_forward(x: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    let last_dim = *shape.last().ok_or(crate::AutogradError::InvalidRank {
        expected: "at least 1",
        got: 0,
    })?;
    let mut out = cpu_sum_last_axis_forward(x, shape)?;
    let inv = 1.0_f32 / last_dim as f32;
    for v in out.iter_mut() {
        *v *= inv;
    }
    Ok(out)
}

/// CPU reference for NeoX RoPE (matches `ops::rope::rope` — element `i` pairs
/// with `i + half_dim`). `x_shape = [batch, heads, seq, head_dim]`; `cos`/`sin`
/// are `[seq, half_dim]` row-major.
pub fn cpu_rope_forward(
    x: &[f32],
    x_shape: &[usize],
    cos: &[f32],
    sin: &[f32],
) -> Result<Vec<f32>> {
    use crate::AutogradError;
    if x_shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: x_shape.len(),
        });
    }
    let batch = x_shape[0];
    let heads = x_shape[1];
    let seq = x_shape[2];
    let head_dim = x_shape[3];
    if !head_dim.is_multiple_of(2) {
        return Err(AutogradError::InvalidRank {
            expected: "even head dim",
            got: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    let expected_x = batch * heads * seq * head_dim;
    if x.len() != expected_x {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected_x],
            got: vec![x.len()],
        });
    }
    if cos.len() != sin.len() || !cos.len().is_multiple_of(seq.max(1)) {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![seq * half_dim],
            got: vec![cos.len().min(sin.len())],
        });
    }
    let rotary_half_dim = cos.len() / seq.max(1);
    if rotary_half_dim == 0 || rotary_half_dim > half_dim {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![seq * half_dim],
            got: vec![cos.len()],
        });
    }
    let rotary_dim = rotary_half_dim * 2;
    let mut out = vec![0.0_f32; expected_x];
    for b in 0..batch {
        for h in 0..heads {
            for t in 0..seq {
                let rope_base = t * rotary_half_dim;
                let base = (((b * heads) + h) * seq + t) * head_dim;
                for i in 0..rotary_half_dim {
                    let x0 = x[base + i];
                    let x1 = x[base + i + rotary_half_dim];
                    let c = cos[rope_base + i];
                    let s = sin[rope_base + i];
                    out[base + i] = (x0 * c) - (x1 * s);
                    out[base + i + rotary_half_dim] = (x1 * c) + (x0 * s);
                }
                out[(base + rotary_dim)..(base + head_dim)]
                    .copy_from_slice(&x[(base + rotary_dim)..(base + head_dim)]);
            }
        }
    }
    Ok(out)
}

/// CPU reference gather along the last axis.
/// `out[prefix] = src[prefix * vocab + ids[prefix]]`. Out-of-range or negative
/// ids produce an error (unlike embedding which zero-fills — the caller is
/// responsible for validating ids).
pub fn cpu_gather_last_dim_forward(
    src: &[f32],
    src_shape: &[usize],
    ids: &[i32],
) -> Result<Vec<f32>> {
    use crate::AutogradError;
    if src_shape.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        });
    }
    let vocab = *src_shape.last().expect("non-empty shape above");
    let prefix: usize = src_shape[..src_shape.len() - 1]
        .iter()
        .product::<usize>()
        .max(1);
    let expected: usize = src_shape.iter().product();
    if src.len() != expected {
        return Err(AutogradError::ShapeMismatch {
            expected: vec![expected],
            got: vec![src.len()],
        });
    }
    if ids.len() != prefix {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix,
            got: ids.len(),
        });
    }
    let mut out = vec![0.0_f32; prefix];
    for (i, &id) in ids.iter().enumerate() {
        if id < 0 || (id as usize) >= vocab {
            return Err(AutogradError::IndexOutOfBounds {
                index: id as usize,
                upper: vocab,
            });
        }
        out[i] = src[i * vocab + id as usize];
    }
    Ok(out)
}

/// CPU reference for `gather_last_dim_backward`. Zero-fills a
/// `src_shape = [prefix..., vocab]` buffer then writes
/// `upstream[row]` into `out[row * vocab + indices[row]]` for each
/// prefix position. Equivalent to the `scatter_add_rows_forward` call
/// in `ops::gather::gather_last_dim_backward` with `feature_dim = 1`
/// and remapped flat ids — kept as a dedicated function so the device
/// backward override returns the same `[B, S, V]` grad shape the
/// autograd graph expects without needing the caller to know about
/// the flat-id trick.
///
/// Negative or out-of-range indices are silently skipped (matches
/// `cpu_scatter_add_rows_forward` and the CUDA kernel's OOB handling).
pub fn cpu_gather_last_dim_backward(
    upstream: &[f32],
    indices: &[i32],
    src_shape: &[usize],
) -> Result<Vec<f32>> {
    use crate::AutogradError;
    if src_shape.is_empty() {
        return Err(AutogradError::InvalidRank {
            expected: "at least 1",
            got: 0,
        });
    }
    let vocab = *src_shape.last().expect("non-empty shape above");
    let prefix: usize = src_shape[..src_shape.len() - 1]
        .iter()
        .product::<usize>()
        .max(1);
    if upstream.len() != prefix {
        return Err(AutogradError::DataLengthMismatch {
            len: upstream.len(),
            shape: src_shape[..src_shape.len() - 1].to_vec(),
            size: prefix,
        });
    }
    if indices.len() != prefix {
        return Err(AutogradError::InvalidIndicesLen {
            expected: prefix,
            got: indices.len(),
        });
    }
    let total = prefix * vocab;
    let mut grad = vec![0.0_f32; total];
    for (row, &id) in indices.iter().enumerate() {
        if id < 0 {
            continue;
        }
        let id_usize = id as usize;
        if id_usize >= vocab {
            continue;
        }
        grad[row * vocab + id_usize] = upstream[row];
    }
    Ok(grad)
}

/// CPU reference scatter-add into a `[vocab, feature_dim]` output.
///
/// `upstream` has length `prefix_rows * feature_dim`; `indices.len() == prefix_rows`.
/// For each row, the feature slice is added into the bin selected by the
/// corresponding index. Negative or out-of-range indices are silently
/// skipped — matches the prior inline scatter in `embedding_backward`
/// (which bounds-checked at the op layer) and the CUDA kernel's OOB
/// handling so behavior is identical across backends.
pub fn cpu_scatter_add_rows_forward(
    upstream: &[f32],
    prefix_rows: usize,
    feature_dim: usize,
    indices: &[i32],
    vocab: usize,
) -> Result<Vec<f32>> {
    let expected_upstream = prefix_rows * feature_dim;
    if upstream.len() != expected_upstream {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![expected_upstream],
            got: vec![upstream.len()],
        });
    }
    if indices.len() != prefix_rows {
        return Err(crate::AutogradError::InvalidIndicesLen {
            expected: prefix_rows,
            got: indices.len(),
        });
    }
    let mut out = vec![0.0_f32; vocab * feature_dim];
    for (row, &id) in indices.iter().enumerate() {
        if id < 0 {
            continue;
        }
        let id = id as usize;
        if id >= vocab {
            continue;
        }
        let src_base = row * feature_dim;
        let dst_base = id * feature_dim;
        for col in 0..feature_dim {
            out[dst_base + col] += upstream[src_base + col];
        }
    }
    Ok(out)
}

/// CPU reference transpose-swap: swap `axis1` and `axis2` of a contiguous
/// row-major tensor with shape `old_shape`. Returns `(data, new_shape)`.
/// Used by the `Backend::transpose_axes_swap` default fallback and by the
/// ops-layer host-eager path — keeping both on the same function means the
/// device-default-fallback and the host path produce byte-identical output
/// for a given input.
pub fn cpu_transpose_swap(
    data: &[f32],
    old_shape: &[usize],
    axis1: usize,
    axis2: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let rank = old_shape.len();
    if axis1 >= rank {
        return Err(crate::AutogradError::AxisOutOfBounds { axis: axis1, rank });
    }
    if axis2 >= rank {
        return Err(crate::AutogradError::AxisOutOfBounds { axis: axis2, rank });
    }
    if axis1 == axis2 {
        return Ok((data.to_vec(), old_shape.to_vec()));
    }

    let mut new_shape = old_shape.to_vec();
    new_shape.swap(axis1, axis2);

    // Contiguous strides over `old_shape` — the source we're reading from.
    let mut old_strides = vec![0usize; rank];
    let mut stride = 1usize;
    for (index, dim) in old_shape.iter().enumerate().rev() {
        old_strides[index] = stride;
        stride *= *dim;
    }

    let mut out = vec![0.0_f32; data.len()];
    for (out_index, slot) in out.iter_mut().enumerate() {
        // Decompose out_index into new_shape coords, then swap the two
        // axes to recover the original source coords.
        let mut coords = vec![0usize; rank];
        let mut linear = out_index;
        for axis in (0..rank).rev() {
            let dim = new_shape[axis];
            coords[axis] = linear % dim;
            linear /= dim;
        }
        coords.swap(axis1, axis2);
        let input_index: usize = coords
            .iter()
            .zip(old_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        *slot = data[input_index];
    }
    Ok((out, new_shape))
}

/// CPU reference contiguous slice: copy elements of `data` (row-major over
/// `old_shape`) whose per-axis coordinate is in `[starts[i], ends[i])`.
/// Returns `(sliced_data, new_shape)` with `new_shape[i] = ends[i] - starts[i]`.
/// Used by the `Backend::slice` default fallback so device-default and host
/// paths share one numerical reference.
pub fn cpu_slice(
    data: &[f32],
    old_shape: &[usize],
    starts: &[usize],
    ends: &[usize],
) -> Result<(Vec<f32>, Vec<usize>)> {
    let rank = old_shape.len();
    let new_shape = validate_slice_shape(old_shape, starts, ends)?;
    let new_numel: usize = if new_shape.is_empty() {
        1
    } else {
        new_shape.iter().product()
    };

    let mut old_strides = vec![0usize; rank];
    let mut stride = 1usize;
    for (index, dim) in old_shape.iter().enumerate().rev() {
        old_strides[index] = stride;
        stride *= *dim;
    }

    let mut out = vec![0.0_f32; new_numel];
    for (out_index, slot) in out.iter_mut().enumerate() {
        let mut coords = vec![0usize; rank];
        let mut linear = out_index;
        for axis in (0..rank).rev() {
            let dim = new_shape[axis];
            if dim > 0 {
                coords[axis] = linear % dim;
                linear /= dim;
            }
        }
        let input_index: usize = coords
            .iter()
            .enumerate()
            .map(|(axis, &c)| (c + starts[axis]) * old_strides[axis])
            .sum();
        *slot = data[input_index];
    }
    Ok((out, new_shape))
}

/// CPU reference for KV-cache append: concatenate two rank-4 contiguous
/// tensors shaped `[batch, heads, seq, dim]` along axis 2.
pub fn cpu_concat_axis2(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>)> {
    if a_shape.len() != 4 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "4",
            got: a_shape.len(),
        });
    }
    if b_shape.len() != 4 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "4",
            got: b_shape.len(),
        });
    }
    if a_shape[0] != b_shape[0] || a_shape[1] != b_shape[1] || a_shape[3] != b_shape[3] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![a_shape[0], a_shape[1], a_shape[3]],
            got: vec![b_shape[0], b_shape[1], b_shape[3]],
        });
    }
    let a_size = shape_size(a_shape);
    if a.len() != a_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: a.len(),
            shape: a_shape.to_vec(),
            size: a_size,
        });
    }
    let b_size = shape_size(b_shape);
    if b.len() != b_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: b.len(),
            shape: b_shape.to_vec(),
            size: b_size,
        });
    }

    let batch = a_shape[0];
    let heads = a_shape[1];
    let a_seq = a_shape[2];
    let b_seq = b_shape[2];
    let dim = a_shape[3];
    let out_shape = vec![batch, heads, a_seq + b_seq, dim];
    let mut out = vec![0.0_f32; shape_size(&out_shape)];

    for batch_idx in 0..batch {
        for head_idx in 0..heads {
            let out_base = ((batch_idx * heads + head_idx) * (a_seq + b_seq)) * dim;
            let a_base = ((batch_idx * heads + head_idx) * a_seq) * dim;
            let b_base = ((batch_idx * heads + head_idx) * b_seq) * dim;
            let a_len = a_seq * dim;
            let b_len = b_seq * dim;
            out[out_base..out_base + a_len].copy_from_slice(&a[a_base..a_base + a_len]);
            out[out_base + a_len..out_base + a_len + b_len]
                .copy_from_slice(&b[b_base..b_base + b_len]);
        }
    }

    Ok((out, out_shape))
}

#[allow(clippy::too_many_arguments)]
pub fn cpu_causal_sdpa_recompute_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    upstream: &[f32],
    shape: &[usize],
    need_grad_q: bool,
    need_grad_k: bool,
    need_grad_v: bool,
) -> Result<CausalSdpaHostGradTriplet> {
    if shape.len() != 4 {
        return Err(AutogradError::InvalidRank {
            expected: "4",
            got: shape.len(),
        });
    }
    let expected = shape_size(shape);
    for len in [q.len(), k.len(), v.len(), upstream.len()] {
        if len != expected {
            return Err(AutogradError::DataLengthMismatch {
                len,
                shape: shape.to_vec(),
                size: expected,
            });
        }
    }

    let batch = shape[0];
    let heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut grad_q = need_grad_q.then(|| vec![0.0; q.len()]);
    let mut grad_k = need_grad_k.then(|| vec![0.0; k.len()]);
    let mut grad_v = need_grad_v.then(|| vec![0.0; v.len()]);
    let mut scores = vec![0.0_f32; seq_len];
    let mut probs = vec![0.0_f32; seq_len];
    let mut d_probs = vec![0.0_f32; seq_len];

    for b in 0..batch {
        for h in 0..heads {
            for row in 0..seq_len {
                let mut max_score = f32::NEG_INFINITY;
                for col in 0..=row {
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += q[offset4(b, h, row, d, heads, seq_len, head_dim)]
                            * k[offset4(b, h, col, d, heads, seq_len, head_dim)];
                    }
                    let score = dot * scale;
                    scores[col] = score;
                    max_score = max_score.max(score);
                }

                let mut denom = 0.0_f32;
                for col in 0..=row {
                    let p = (scores[col] - max_score).exp();
                    probs[col] = p;
                    denom += p;
                }
                for prob in probs.iter_mut().take(row + 1) {
                    *prob /= denom;
                }

                for col in 0..=row {
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += upstream[offset4(b, h, row, d, heads, seq_len, head_dim)]
                            * v[offset4(b, h, col, d, heads, seq_len, head_dim)];
                    }
                    d_probs[col] = dot;
                }

                let mut softmax_dot = 0.0_f32;
                for col in 0..=row {
                    softmax_dot += d_probs[col] * probs[col];
                }

                for col in 0..=row {
                    let d_score = probs[col] * (d_probs[col] - softmax_dot);
                    if let Some(grad_v) = grad_v.as_mut() {
                        for d in 0..head_dim {
                            grad_v[offset4(b, h, col, d, heads, seq_len, head_dim)] += probs[col]
                                * upstream[offset4(b, h, row, d, heads, seq_len, head_dim)];
                        }
                    }
                    if let Some(grad_q) = grad_q.as_mut() {
                        for d in 0..head_dim {
                            grad_q[offset4(b, h, row, d, heads, seq_len, head_dim)] += scale
                                * d_score
                                * k[offset4(b, h, col, d, heads, seq_len, head_dim)];
                        }
                    }
                    if let Some(grad_k) = grad_k.as_mut() {
                        for d in 0..head_dim {
                            grad_k[offset4(b, h, col, d, heads, seq_len, head_dim)] += scale
                                * d_score
                                * q[offset4(b, h, row, d, heads, seq_len, head_dim)];
                        }
                    }
                }
            }
        }
    }

    Ok((grad_q, grad_k, grad_v))
}

#[inline]
fn offset4(
    batch: usize,
    head: usize,
    token: usize,
    dim: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> usize {
    (((batch * heads + head) * seq_len + token) * head_dim) + dim
}

/// CPU reference for decode-time GQA causal attention.
#[allow(clippy::too_many_arguments)]
pub fn cpu_causal_sdpa_decode_gqa(
    q: &[f32],
    q_shape: &[usize],
    k: &[f32],
    k_shape: &[usize],
    v: &[f32],
    v_shape: &[usize],
    q_start: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    validate_decode_gqa_shapes(q_shape, k_shape, v_shape, q_start)?;
    let q_size = shape_size(q_shape);
    let k_size = shape_size(k_shape);
    let v_size = shape_size(v_shape);
    if q.len() != q_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: q.len(),
            shape: q_shape.to_vec(),
            size: q_size,
        });
    }
    if k.len() != k_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: k.len(),
            shape: k_shape.to_vec(),
            size: k_size,
        });
    }
    if v.len() != v_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: v.len(),
            shape: v_shape.to_vec(),
            size: v_size,
        });
    }

    let batch = q_shape[0];
    let query_heads = q_shape[1];
    let kv_heads = k_shape[1];
    let kv_len = k_shape[2];
    let head_dim = q_shape[3];
    let kv_repeat = query_heads / kv_heads;
    let visible = (q_start + 1).min(kv_len);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let out_shape = vec![batch, query_heads, 1, head_dim];
    let mut out = vec![0.0_f32; shape_size(&out_shape)];
    let mut scores = vec![0.0_f32; visible];

    for batch_idx in 0..batch {
        for query_head in 0..query_heads {
            let kv_head = query_head / kv_repeat;
            let q_base = (batch_idx * query_heads + query_head) * head_dim;

            let mut max_score = f32::NEG_INFINITY;
            for (pos, score_slot) in scores.iter_mut().enumerate().take(visible) {
                let k_base = ((batch_idx * kv_heads + kv_head) * kv_len + pos) * head_dim;
                let mut dot = 0.0_f32;
                for dim in 0..head_dim {
                    dot += q[q_base + dim] * k[k_base + dim];
                }
                let score = dot * scale;
                *score_slot = score;
                max_score = max_score.max(score);
            }

            let mut denom = 0.0_f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denom += *score;
            }
            let out_base = (batch_idx * query_heads + query_head) * head_dim;
            if denom == 0.0 {
                continue;
            }

            for dim in 0..head_dim {
                let mut acc = 0.0_f32;
                for (pos, &weight_exp) in scores.iter().enumerate() {
                    let v_base = ((batch_idx * kv_heads + kv_head) * kv_len + pos) * head_dim;
                    acc += (weight_exp / denom) * v[v_base + dim];
                }
                out[out_base + dim] = acc;
            }
        }
    }

    Ok((out, out_shape))
}

pub fn cpu_kv_cache_write_axis2(
    dst: &mut [f32],
    dst_shape: &[usize],
    src: &[f32],
    src_shape: &[usize],
    seq_offset: usize,
) -> Result<()> {
    for shape in [dst_shape, src_shape] {
        if shape.len() != 4 {
            return Err(crate::AutogradError::InvalidRank {
                expected: "4",
                got: shape.len(),
            });
        }
    }
    if dst_shape[0] != src_shape[0] || dst_shape[1] != src_shape[1] || dst_shape[3] != src_shape[3]
    {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![dst_shape[0], dst_shape[1], dst_shape[3]],
            got: vec![src_shape[0], src_shape[1], src_shape[3]],
        });
    }
    let src_seq = src_shape[2];
    if seq_offset + src_seq > dst_shape[2] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![dst_shape[2]],
            got: vec![seq_offset + src_seq],
        });
    }
    let dst_size = shape_size(dst_shape);
    let src_size = shape_size(src_shape);
    if dst.len() != dst_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: dst.len(),
            shape: dst_shape.to_vec(),
            size: dst_size,
        });
    }
    if src.len() != src_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: src.len(),
            shape: src_shape.to_vec(),
            size: src_size,
        });
    }

    let batch = dst_shape[0];
    let heads = dst_shape[1];
    let max_seq = dst_shape[2];
    let dim = dst_shape[3];
    for batch_idx in 0..batch {
        for head_idx in 0..heads {
            for seq_idx in 0..src_seq {
                let src_base = ((batch_idx * heads + head_idx) * src_seq + seq_idx) * dim;
                let dst_base =
                    ((batch_idx * heads + head_idx) * max_seq + seq_offset + seq_idx) * dim;
                dst[dst_base..dst_base + dim].copy_from_slice(&src[src_base..src_base + dim]);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn cpu_causal_sdpa_decode_gqa_cache(
    q: &[f32],
    q_shape: &[usize],
    k: &[f32],
    k_shape: &[usize],
    v: &[f32],
    v_shape: &[usize],
    kv_len: usize,
    q_start: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    validate_decode_gqa_cache_shapes(q_shape, k_shape, v_shape, kv_len, q_start)?;
    let q_size = shape_size(q_shape);
    let k_size = shape_size(k_shape);
    let v_size = shape_size(v_shape);
    if q.len() != q_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: q.len(),
            shape: q_shape.to_vec(),
            size: q_size,
        });
    }
    if k.len() != k_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: k.len(),
            shape: k_shape.to_vec(),
            size: k_size,
        });
    }
    if v.len() != v_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: v.len(),
            shape: v_shape.to_vec(),
            size: v_size,
        });
    }

    let batch = q_shape[0];
    let query_heads = q_shape[1];
    let kv_heads = k_shape[1];
    let max_seq = k_shape[2];
    let head_dim = q_shape[3];
    let kv_repeat = query_heads / kv_heads;
    let visible = (q_start + 1).min(kv_len);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let out_shape = vec![batch, query_heads, 1, head_dim];
    let mut out = vec![0.0_f32; shape_size(&out_shape)];
    let mut scores = vec![0.0_f32; visible];

    for batch_idx in 0..batch {
        for query_head in 0..query_heads {
            let kv_head = query_head / kv_repeat;
            let q_base = (batch_idx * query_heads + query_head) * head_dim;

            let mut max_score = f32::NEG_INFINITY;
            for (pos, score_slot) in scores.iter_mut().enumerate().take(visible) {
                let k_base = ((batch_idx * kv_heads + kv_head) * max_seq + pos) * head_dim;
                let mut dot = 0.0_f32;
                for dim in 0..head_dim {
                    dot += q[q_base + dim] * k[k_base + dim];
                }
                let score = dot * scale;
                *score_slot = score;
                max_score = max_score.max(score);
            }

            let mut denom = 0.0_f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denom += *score;
            }
            if denom == 0.0 {
                continue;
            }

            let out_base = (batch_idx * query_heads + query_head) * head_dim;
            for dim in 0..head_dim {
                let mut acc = 0.0_f32;
                for (pos, &weight_exp) in scores.iter().enumerate() {
                    let v_base = ((batch_idx * kv_heads + kv_head) * max_seq + pos) * head_dim;
                    acc += (weight_exp / denom) * v[v_base + dim];
                }
                out[out_base + dim] = acc;
            }
        }
    }
    Ok((out, out_shape))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_qwen_decode_prepare_q(
    q_full: &[f32],
    q_full_shape: &[usize],
    q_norm_weight: &[f32],
    q_norm_weight_shape: &[usize],
    cos: &[f32],
    cos_shape: &[usize],
    sin: &[f32],
    sin_shape: &[usize],
    query_heads: usize,
    head_dim: usize,
    gated: bool,
    eps: f32,
) -> Result<QwenDecodePrepareQHost> {
    validate_qwen_decode_prepare_q_shapes(
        q_full_shape,
        q_norm_weight_shape,
        cos_shape,
        sin_shape,
        query_heads,
        head_dim,
        gated,
    )?;
    let q_full_size = shape_size(q_full_shape);
    if q_full.len() != q_full_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: q_full.len(),
            shape: q_full_shape.to_vec(),
            size: q_full_size,
        });
    }
    if q_norm_weight.len() != head_dim {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: q_norm_weight.len(),
            shape: q_norm_weight_shape.to_vec(),
            size: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    if cos.len() != half_dim || sin.len() != half_dim {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: cos.len().max(sin.len()),
            shape: vec![1, half_dim],
            size: half_dim,
        });
    }

    let batch = q_full_shape[0];
    let q_full_stride = q_full_shape[2];
    let head_stride = if gated { head_dim * 2 } else { head_dim };
    let out_shape = vec![batch, query_heads, 1, head_dim];
    let mut q_layout = vec![0.0_f32; shape_size(&out_shape)];
    let mut gate_layout = gated.then(|| vec![0.0_f32; shape_size(&out_shape)]);

    for batch_idx in 0..batch {
        for head in 0..query_heads {
            let src_base = batch_idx * q_full_stride + head * head_stride;
            let out_base = (batch_idx * query_heads + head) * head_dim;
            q_layout[out_base..out_base + head_dim]
                .copy_from_slice(&q_full[src_base..src_base + head_dim]);
            if let Some(gate) = gate_layout.as_mut() {
                gate[out_base..out_base + head_dim]
                    .copy_from_slice(&q_full[src_base + head_dim..src_base + head_stride]);
            }
        }
    }

    let q_norm_weight: Vec<f32> = q_norm_weight.iter().map(|&value| value + 1.0).collect();
    let q_normed = cpu_rms_norm_forward(&q_layout, &q_norm_weight, &out_shape, eps)?;
    let q_roped = cpu_rope_forward(&q_normed, &out_shape, cos, sin)?;
    Ok((q_roped, gate_layout, out_shape))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_qwen_decode_prepare_kv(
    k_full: &[f32],
    k_full_shape: &[usize],
    v_full: &[f32],
    v_full_shape: &[usize],
    k_norm_weight: &[f32],
    k_norm_weight_shape: &[usize],
    cos: &[f32],
    cos_shape: &[usize],
    sin: &[f32],
    sin_shape: &[usize],
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<QwenDecodePrepareKvHost> {
    validate_qwen_decode_prepare_kv_shapes(
        k_full_shape,
        v_full_shape,
        k_norm_weight_shape,
        cos_shape,
        sin_shape,
        kv_heads,
        head_dim,
    )?;
    let k_full_size = shape_size(k_full_shape);
    let v_full_size = shape_size(v_full_shape);
    if k_full.len() != k_full_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: k_full.len(),
            shape: k_full_shape.to_vec(),
            size: k_full_size,
        });
    }
    if v_full.len() != v_full_size {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: v_full.len(),
            shape: v_full_shape.to_vec(),
            size: v_full_size,
        });
    }
    if k_norm_weight.len() != head_dim {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: k_norm_weight.len(),
            shape: k_norm_weight_shape.to_vec(),
            size: head_dim,
        });
    }
    let half_dim = head_dim / 2;
    if cos.len() != half_dim || sin.len() != half_dim {
        return Err(crate::AutogradError::DataLengthMismatch {
            len: cos.len().max(sin.len()),
            shape: vec![1, half_dim],
            size: half_dim,
        });
    }

    let batch = k_full_shape[0];
    let full_stride = k_full_shape[2];
    let out_shape = vec![batch, kv_heads, 1, head_dim];
    let mut k_layout = vec![0.0_f32; shape_size(&out_shape)];
    let mut v_layout = vec![0.0_f32; shape_size(&out_shape)];

    for batch_idx in 0..batch {
        for head in 0..kv_heads {
            let src_base = batch_idx * full_stride + head * head_dim;
            let out_base = (batch_idx * kv_heads + head) * head_dim;
            k_layout[out_base..out_base + head_dim]
                .copy_from_slice(&k_full[src_base..src_base + head_dim]);
            v_layout[out_base..out_base + head_dim]
                .copy_from_slice(&v_full[src_base..src_base + head_dim]);
        }
    }

    let k_norm_weight: Vec<f32> = k_norm_weight.iter().map(|&value| value + 1.0).collect();
    let k_normed = cpu_rms_norm_forward(&k_layout, &k_norm_weight, &out_shape, eps)?;
    let k_roped = cpu_rope_forward(&k_normed, &out_shape, cos, sin)?;
    Ok((k_roped, v_layout, out_shape))
}

fn validate_qwen_decode_prepare_common(
    full_shape: &[usize],
    weight_shape: &[usize],
    cos_shape: &[usize],
    sin_shape: &[usize],
    heads: usize,
    head_dim: usize,
    projected_dim: usize,
) -> Result<()> {
    if full_shape.len() != 3 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "3",
            got: full_shape.len(),
        });
    }
    if full_shape[1] != 1 || full_shape[2] != projected_dim {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![full_shape[0], 1, projected_dim],
            got: full_shape.to_vec(),
        });
    }
    if heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(crate::AutogradError::TapeInvariant(
            "qwen decode prepare requires non-zero heads and even head_dim",
        ));
    }
    if weight_shape != [head_dim] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![head_dim],
            got: weight_shape.to_vec(),
        });
    }
    let rope_shape = [1, head_dim / 2];
    if cos_shape != rope_shape || sin_shape != rope_shape {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: rope_shape.to_vec(),
            got: cos_shape.to_vec(),
        });
    }
    Ok(())
}

pub(crate) fn validate_qwen_decode_prepare_q_shapes(
    q_full_shape: &[usize],
    q_norm_weight_shape: &[usize],
    cos_shape: &[usize],
    sin_shape: &[usize],
    query_heads: usize,
    head_dim: usize,
    gated: bool,
) -> Result<()> {
    let factor = if gated { 2 } else { 1 };
    validate_qwen_decode_prepare_common(
        q_full_shape,
        q_norm_weight_shape,
        cos_shape,
        sin_shape,
        query_heads,
        head_dim,
        query_heads * head_dim * factor,
    )
}

pub(crate) fn validate_qwen_decode_prepare_kv_shapes(
    k_full_shape: &[usize],
    v_full_shape: &[usize],
    k_norm_weight_shape: &[usize],
    cos_shape: &[usize],
    sin_shape: &[usize],
    kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    validate_qwen_decode_prepare_common(
        k_full_shape,
        k_norm_weight_shape,
        cos_shape,
        sin_shape,
        kv_heads,
        head_dim,
        kv_heads * head_dim,
    )?;
    if v_full_shape != k_full_shape {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: k_full_shape.to_vec(),
            got: v_full_shape.to_vec(),
        });
    }
    Ok(())
}

pub fn validate_decode_gqa_shapes(
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
    q_start: usize,
) -> Result<()> {
    for shape in [q_shape, k_shape, v_shape] {
        if shape.len() != 4 {
            return Err(crate::AutogradError::InvalidRank {
                expected: "4",
                got: shape.len(),
            });
        }
    }

    if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if q_shape[2] != 1 {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![1],
            got: vec![q_shape[2]],
        });
    }
    if q_shape[3] != k_shape[3] || q_shape[3] != v_shape[3] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if k_shape[1] != v_shape[1] || k_shape[2] != v_shape[2] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: k_shape.to_vec(),
            got: v_shape.to_vec(),
        });
    }
    if k_shape[2] == 0 {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-empty kv_len",
            got: 0,
        });
    }
    if q_shape[1] == 0 || k_shape[1] == 0 || !q_shape[1].is_multiple_of(k_shape[1]) {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![q_shape[1]],
            got: vec![k_shape[1]],
        });
    }
    if q_start >= k_shape[2] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![q_start + 1],
            got: vec![k_shape[2]],
        });
    }

    Ok(())
}

pub fn validate_decode_gqa_cache_shapes(
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
    kv_len: usize,
    q_start: usize,
) -> Result<()> {
    for shape in [q_shape, k_shape, v_shape] {
        if shape.len() != 4 {
            return Err(crate::AutogradError::InvalidRank {
                expected: "4",
                got: shape.len(),
            });
        }
    }

    if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if q_shape[2] != 1 {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![1],
            got: vec![q_shape[2]],
        });
    }
    if q_shape[3] != k_shape[3] || q_shape[3] != v_shape[3] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: q_shape.to_vec(),
            got: k_shape.to_vec(),
        });
    }
    if k_shape[1] != v_shape[1] || k_shape[2] != v_shape[2] {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: k_shape.to_vec(),
            got: v_shape.to_vec(),
        });
    }
    if k_shape[2] == 0 || kv_len == 0 || kv_len > k_shape[2] {
        return Err(crate::AutogradError::InvalidRank {
            expected: "non-empty kv_len within cache capacity",
            got: kv_len,
        });
    }
    if q_shape[1] == 0 || k_shape[1] == 0 || !q_shape[1].is_multiple_of(k_shape[1]) {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![q_shape[1]],
            got: vec![k_shape[1]],
        });
    }
    if q_start >= kv_len {
        return Err(crate::AutogradError::ShapeMismatch {
            expected: vec![q_start + 1],
            got: vec![kv_len],
        });
    }

    Ok(())
}

pub(crate) fn validate_slice_shape(
    old_shape: &[usize],
    starts: &[usize],
    ends: &[usize],
) -> Result<Vec<usize>> {
    let rank = old_shape.len();
    if starts.len() != rank {
        return Err(crate::AutogradError::InvalidIndicesLen {
            expected: rank,
            got: starts.len(),
        });
    }
    if ends.len() != rank {
        return Err(crate::AutogradError::InvalidIndicesLen {
            expected: rank,
            got: ends.len(),
        });
    }
    for ((&start, &end), &dim) in starts.iter().zip(ends.iter()).zip(old_shape.iter()) {
        if start > end {
            return Err(crate::AutogradError::TapeInvariant(
                "cpu_slice: start must be <= end for every axis",
            ));
        }
        if end > dim {
            return Err(crate::AutogradError::IndexOutOfBounds {
                index: end,
                upper: dim,
            });
        }
        if start > dim {
            return Err(crate::AutogradError::IndexOutOfBounds {
                index: start,
                upper: dim,
            });
        }
    }
    Ok(starts
        .iter()
        .zip(ends.iter())
        .map(|(&start, &end)| end - start)
        .collect())
}

/// CPU reference in-place AdamW update. Matches the formula in
/// `optim.rs::AdamW::step`:
///
/// - weight decay (decoupled): `param *= 1 - lr * wd`
/// - EMAs: `m = β1·m + (1-β1)·g`, `v = β2·v + (1-β2)·g²`
/// - bias-corrected step: `param -= lr · (m/bc1) / (√(v/bc2) + eps)`
///
/// Exposed as a free fn so `Backend::adamw_step` default impl and the
/// optimizer's host path share one numerical reference. Any backend
/// override (e.g. `MetalBackend::adamw_step`) MUST match this to the
/// 1e-5 gate enforced by `metal_adamw_step_stays_device_resident`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_adamw_step_in_place(
    param: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    grad: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    bc1: f32,
    bc2: f32,
) {
    debug_assert_eq!(param.len(), m.len());
    debug_assert_eq!(param.len(), v.len());
    debug_assert_eq!(param.len(), grad.len());

    if wd > 0.0 {
        let decay = 1.0 - (lr * wd);
        for value in param.iter_mut() {
            *value *= decay;
        }
    }

    for index in 0..param.len() {
        let g = grad[index];
        m[index] = (beta1 * m[index]) + ((1.0 - beta1) * g);
        v[index] = (beta2 * v[index]) + ((1.0 - beta2) * g * g);
        let m_hat = m[index] / bc1;
        let v_hat = v[index] / bc2;
        param[index] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

pub(crate) fn matmul_output_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    use crate::AutogradError;

    match (a_shape.len(), b_shape.len()) {
        (2, 2) => {
            if a_shape[1] != b_shape[0] {
                return Err(AutogradError::ShapeMismatch {
                    expected: vec![a_shape[1]],
                    got: vec![b_shape[0]],
                });
            }
            Ok(vec![a_shape[0], b_shape[1]])
        }
        (3, 3) => {
            if a_shape[0] != b_shape[0] {
                return Err(AutogradError::ShapeMismatch {
                    expected: vec![a_shape[0]],
                    got: vec![b_shape[0]],
                });
            }
            if a_shape[2] != b_shape[1] {
                return Err(AutogradError::ShapeMismatch {
                    expected: vec![a_shape[2]],
                    got: vec![b_shape[1]],
                });
            }
            Ok(vec![a_shape[0], a_shape[1], b_shape[2]])
        }
        _ => Err(AutogradError::InvalidRank {
            expected: "both operands must be rank-2 or rank-3",
            got: a_shape.len().max(b_shape.len()),
        }),
    }
}
