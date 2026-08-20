//! Dense quantized-linear launch helpers (FP8 family).

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi::{self, Half};
use crate::tensor::{DeviceContext, RawDevicePtr};

// Safe wrappers over the dense FP8 quant-linear FFI, per the `moe.rs` pattern:
// typed buffers, checked i32 casts, pointer guards held through submission, one
// FFI symbol per launcher. bf16 (Rust) and Half (u16, kernel ABI) share a
// 16-bit layout, so pointers cast directly.

/// FP8 f32 scale-grid metadata as the block-scaled kernels consume it:
/// `scales[(row/block_m)*scale_cols + (col/block_k)]`.
#[derive(Clone, Copy, Debug)]
pub struct Fp8ScaleShape {
    pub scale_rows: i32,
    pub scale_cols: i32,
    pub block_m: i32,
    pub block_k: i32,
}

impl Fp8ScaleShape {
    fn grid_len(self) -> usize {
        self.scale_rows.max(0) as usize * self.scale_cols.max(0) as usize
    }
}

fn extent(a: usize, b: usize, what: &'static str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| anyhow!("{what} shape overflow: {a}x{b}"))
}

/// Per-channel FP8 Marlin GEMM: C[m,n] = X[m,k] @ dequant(Marlin-packed W).
/// `c_tmp` / `workspace` are the shared Marlin scratch, sized to the SM max at
/// alloc and mutated by the kernel (the workspace locks return to 0 after each
/// GEMM), hence the mut casts off shared borrows.
#[allow(clippy::too_many_arguments)]
pub fn marlin_fp8_gemm(
    ctx: &DeviceContext,
    input: &impl DevicePtr<bf16>,
    packed: &impl DevicePtr<u8>,
    scales: &impl DevicePtr<u16>,
    output: &mut impl DevicePtrMut<bf16>,
    c_tmp: &impl DevicePtr<f32>,
    workspace: &impl DevicePtr<i32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    ensure!(
        input.len() >= extent(m, k, "marlin_fp8_gemm input")?
            && output.len() >= extent(m, n, "marlin_fp8_gemm output")?,
        "marlin_fp8_gemm buffers do not cover [m,n,k]=[{m},{n},{k}]: input={} output={}",
        input.len(),
        output.len()
    );
    let (x_ptr, _gx) = input.device_ptr(&ctx.stream);
    let (packed_ptr, _gp) = packed.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    let (c_tmp_ptr, _gc) = c_tmp.device_ptr(&ctx.stream);
    let (ws_ptr, _gw) = workspace.device_ptr(&ctx.stream);
    // SAFETY: lengths checked above; the packed bytes are the u32 Marlin tiles
    // `repack_for_marlin_fp8` produced for these dims.
    unsafe {
        ffi::marlin_fp8_gemm_cuda(
            x_ptr as *const Half,
            packed_ptr as *const u32,
            scales_ptr as *const Half,
            out_ptr as *mut Half,
            c_tmp_ptr as *mut f32,
            ws_ptr as *mut i32,
            i32::try_from(m)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("marlin_fp8_gemm_cuda failed at [m,n,k]=[{m},{n},{k}]: {e}"))
    }
}

/// Scalar block-scaled FP8 GEMV, single row: out[n] = W[n,k] @ x[k].
pub fn gemv_fp8_block_scaled(
    ctx: &DeviceContext,
    weight: &impl DevicePtr<u8>,
    scales: &impl DevicePtr<f32>,
    input: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
    n: usize,
    k: usize,
    scale: Fp8ScaleShape,
) -> Result<()> {
    ensure!(
        weight.len() >= extent(n, k, "gemv_fp8_block_scaled weight")?
            && scales.len() >= scale.grid_len()
            && input.len() >= k
            && output.len() >= n,
        "gemv_fp8_block_scaled buffers do not cover [n,k]=[{n},{k}]: weight={} scales={} input={} output={}",
        weight.len(),
        scales.len(),
        input.len(),
        output.len()
    );
    let (qw_ptr, _gqw) = weight.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = input.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::gemv_fp8_block_scaled_cuda(
            qw_ptr as *const u8,
            scales_ptr as *const f32,
            x_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(n)?,
            i32::try_from(k)?,
            scale.scale_rows,
            scale.scale_cols,
            scale.block_m,
            scale.block_k,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("gemv_fp8_block_scaled_cuda failed at [n,k]=[{n},{k}]: {e}"))
    }
}

/// Batched block-scaled FP8 GEMV: out[m,n] = X[m,k] @ W[n,k]^T.
#[allow(clippy::too_many_arguments)]
pub fn gemv_fp8_block_scaled_batch(
    ctx: &DeviceContext,
    weight: &impl DevicePtr<u8>,
    scales: &impl DevicePtr<f32>,
    input: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
    m: usize,
    n: usize,
    k: usize,
    scale: Fp8ScaleShape,
) -> Result<()> {
    ensure!(
        weight.len() >= extent(n, k, "gemv_fp8_block_scaled_batch weight")?
            && scales.len() >= scale.grid_len()
            && input.len() >= extent(m, k, "gemv_fp8_block_scaled_batch input")?
            && output.len() >= extent(m, n, "gemv_fp8_block_scaled_batch output")?,
        "gemv_fp8_block_scaled_batch buffers do not cover [m,n,k]=[{m},{n},{k}]: weight={} scales={} input={} output={}",
        weight.len(),
        scales.len(),
        input.len(),
        output.len()
    );
    let (qw_ptr, _gqw) = weight.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = input.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::gemv_fp8_block_scaled_batch_cuda(
            qw_ptr as *const u8,
            scales_ptr as *const f32,
            x_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(m)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            scale.scale_rows,
            scale.scale_cols,
            scale.block_m,
            scale.block_k,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("gemv_fp8_block_scaled_batch_cuda failed at [m,n,k]=[{m},{n},{k}]: {e}")
        })
    }
}

/// Software-dequant an FP8 E4M3 block-scaled weight `[n, k]` into a resident
/// BF16 buffer.
///
/// # Safety
/// `output` must cover `n * k` bf16 elements on `ctx`'s stream and stay live
/// through the launch (it is the caller's reusable scratch).
pub unsafe fn dequantize_fp8_block_scaled_to_bf16(
    ctx: &DeviceContext,
    weight: &impl DevicePtr<u8>,
    scales: &impl DevicePtr<f32>,
    output: RawDevicePtr<bf16>,
    n: usize,
    k: usize,
    scale: Fp8ScaleShape,
) -> Result<()> {
    ensure!(
        weight.len() >= extent(n, k, "dequantize_fp8_block_scaled weight")?
            && scales.len() >= scale.grid_len(),
        "dequantize_fp8_block_scaled buffers do not cover [n,k]=[{n},{k}]: weight={} scales={}",
        weight.len(),
        scales.len()
    );
    let (qw_ptr, _gqw) = weight.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    // SAFETY: source lengths checked above; the output scratch is the caller's
    // contract.
    unsafe {
        ffi::dequantize_fp8_block_scaled_to_bf16_cuda(
            qw_ptr as *const u8,
            scales_ptr as *const f32,
            output.as_mut_ptr().cast(),
            i32::try_from(n)?,
            i32::try_from(k)?,
            scale.scale_rows,
            scale.scale_cols,
            scale.block_m,
            scale.block_k,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("dequantize_fp8_block_scaled_to_bf16_cuda failed at [n,k]=[{n},{k}]: {e}")
        })
    }
}

/// BF16 GEMM against a dequantized weight: C[m,n] = X[m,k] @ W[n,k]^T
/// (`gemm_cuda`'s M is the weight-row dim n).
///
/// # Safety
/// `weight` must cover `n * k` bf16 elements on `ctx`'s stream and stay live
/// through the launch (it is the caller's reusable scratch).
pub unsafe fn gemm_bf16(
    ctx: &DeviceContext,
    weight: RawDevicePtr<bf16>,
    input: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    ensure!(
        input.len() >= extent(m, k, "gemm_bf16 input")?
            && output.len() >= extent(m, n, "gemm_bf16 output")?,
        "gemm_bf16 buffers do not cover [m,n,k]=[{m},{n},{k}]: input={} output={}",
        input.len(),
        output.len()
    );
    let (x_ptr, _gx) = input.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    // SAFETY: activation/output lengths checked above; the weight scratch is
    // the caller's contract.
    unsafe {
        ffi::gemm_cuda(
            weight.as_ptr().cast(),
            x_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(n)?,
            i32::try_from(m)?,
            i32::try_from(k)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("gemm_cuda failed at [m,n,k]=[{m},{n},{k}]: {e}"))
    }
}

/// Un-permute Marlin per-channel FP8 tiles back to the plain `[n, k]` E4M3
/// bytes DeepGEMM's dense NT entry takes as B.
///
/// # Safety
/// `output` must cover `n * k` bytes on `ctx`'s stream and stay live through
/// the launch (it is the caller's reusable scratch).
pub unsafe fn marlin_fp8_to_e4m3(
    ctx: &DeviceContext,
    packed: &impl DevicePtr<u8>,
    output: RawDevicePtr<u8>,
    n: usize,
    k: usize,
) -> Result<()> {
    ensure!(
        packed.len() >= extent(n, k, "marlin_fp8_to_e4m3 packed")?,
        "marlin_fp8_to_e4m3 packed tiles do not cover [n,k]=[{n},{k}]: packed={}",
        packed.len()
    );
    let (packed_ptr, _gp) = packed.device_ptr(&ctx.stream);
    // SAFETY: packed length checked above; the output scratch is the caller's
    // contract.
    unsafe {
        ffi::marlin_fp8_to_e4m3_cuda(
            packed_ptr as *const u8,
            output.as_mut_ptr(),
            i32::try_from(n)?,
            i32::try_from(k)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("marlin_fp8_to_e4m3_cuda failed at [n,k]=[{n},{k}]: {e}"))
    }
}

/// In-place per-column scale over a bf16 `[rows, cols]` buffer:
/// `data[r, c] *= scales[c]`.
pub fn scale_columns_bf16(
    ctx: &DeviceContext,
    data: &mut impl DevicePtrMut<bf16>,
    scales: &impl DevicePtr<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    ensure!(
        data.len() >= extent(rows, cols, "scale_columns_bf16 data")? && scales.len() >= cols,
        "scale_columns_bf16 buffers do not cover [rows,cols]=[{rows},{cols}]: data={} scales={}",
        data.len(),
        scales.len()
    );
    let (data_ptr, _gd) = data.device_ptr_mut(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    // SAFETY: lengths checked above; both buffers belong to `ctx.stream`.
    unsafe {
        ffi::scale_columns_bf16_cuda(
            data_ptr as *mut Half,
            scales_ptr as *const f32,
            i32::try_from(rows)?,
            i32::try_from(cols)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("scale_columns_bf16_cuda failed at [rows,cols]=[{rows},{cols}]: {e}"))
    }
}
