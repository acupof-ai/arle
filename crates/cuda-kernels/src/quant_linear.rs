//! Dense quantized-linear launch helpers (FP8, NVFP4, and INT families).

use anyhow::{Result, anyhow, ensure};
use cudarc::driver::sys::{CUresult, CUstream};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi::{self, Half};
use crate::tensor::{DeviceContext, RawDevicePtr, cache_ptr};

// bf16 (Rust) and Half (u16, kernel ABI) share a 16-bit layout, so pointers cast directly.

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

/// Dequantize a per-channel FP8 Marlin-packed weight to dense BF16. The tiles
/// reproduce the checkpoint's own E4M3 bytes (`marlin_fp8_to_e4m3`), then the
/// held per-channel scales apply (`dequantize_fp8_block_scaled_to_bf16` with a
/// 1×K block shape). Composes the two existing kernels; the E4M3 scratch is
/// freed on return. This is the FP8 analog of `dequantize_fp4_marlin_to_bf16`.
///
/// # Safety
/// `packed` must be the layout `repack_for_marlin_fp8` produced, and `output`
/// must cover n*k bf16 on this stream (caller's contract).
pub unsafe fn dequantize_fp8_marlin_to_bf16(
    ctx: &DeviceContext,
    packed: &impl DevicePtr<u8>,
    scales: &impl DevicePtr<f32>,
    output: RawDevicePtr<bf16>,
    n: usize,
    k: usize,
) -> Result<()> {
    let nk = extent(n, k, "dequantize_fp8_marlin_to_bf16 weight")?;
    ensure!(
        packed.len() >= nk && scales.len() >= n,
        "dequantize_fp8_marlin_to_bf16 buffers do not cover [n,k]=[{n},{k}]: packed={} scales={}",
        packed.len(),
        scales.len()
    );
    let scratch = ctx
        .stream
        .alloc_zeros::<u8>(nk)
        .map_err(|e| anyhow!("dequantize_fp8_marlin_to_bf16 scratch alloc: {e}"))?;
    // SAFETY: lengths checked above; `packed` is the layout `repack_for_marlin_fp8`
    // produced, and `output` covers n*k bf16 on this stream (caller's contract).
    unsafe {
        marlin_fp8_to_e4m3(ctx, packed, cache_ptr(&scratch, ctx), n, k)
            .map_err(|e| anyhow!("marlin_fp8_to_e4m3 in dequantize_fp8_marlin_to_bf16: {e}"))?;
        dequantize_fp8_block_scaled_to_bf16(
            ctx,
            &scratch,
            scales,
            output,
            n,
            k,
            Fp8ScaleShape {
                scale_rows: i32::try_from(n)?,
                scale_cols: 1,
                block_m: 1,
                block_k: i32::try_from(k)?,
            },
        )
        .map_err(|e| {
            anyhow!("dequantize_fp8_block_scaled_to_bf16 in dequantize_fp8_marlin_to_bf16: {e}")
        })
    }
}

/// NVFP4 Marlin GEMM: C[m,n] = X[m,k] @ dequant(Marlin-packed W). The S0E5M3
/// group scales sit in the tail of `packed`, after the `n*k/2` tile bytes
/// (`repack_for_marlin_fp4`); `global` is the single BF16 that carries the
/// per-tensor scale times the dequant bias. Scratch contract matches
/// `marlin_fp8_gemm`.
#[allow(clippy::too_many_arguments)]
pub fn marlin_fp4_gemm(
    ctx: &DeviceContext,
    input: &impl DevicePtr<bf16>,
    packed: &impl DevicePtr<u8>,
    global: &impl DevicePtr<u16>,
    output: &mut impl DevicePtrMut<bf16>,
    c_tmp: &impl DevicePtr<f32>,
    workspace: &impl DevicePtr<i32>,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<()> {
    let nk = extent(n, k, "marlin_fp4_gemm weight")?;
    ensure!(
        group_size > 0
            && packed.len() >= nk / 2 + nk / group_size
            && input.len() >= extent(m, k, "marlin_fp4_gemm input")?
            && output.len() >= extent(m, n, "marlin_fp4_gemm output")?,
        "marlin_fp4_gemm buffers do not cover [m,n,k,gs]=[{m},{n},{k},{group_size}]: packed={} input={} output={}",
        packed.len(),
        input.len(),
        output.len()
    );
    let (x_ptr, _gx) = input.device_ptr(&ctx.stream);
    let (packed_ptr, _gp) = packed.device_ptr(&ctx.stream);
    // The S0E5M3 group scales sit in the tail of the same allocation; see
    // `repack_for_marlin_fp4`.
    let scales_ptr = packed_ptr + (nk / 2) as u64;
    let (global_ptr, _gg) = global.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    let (c_tmp_ptr, _gc) = c_tmp.device_ptr(&ctx.stream);
    let (ws_ptr, _gw) = workspace.device_ptr(&ctx.stream);
    // SAFETY: lengths checked above; packed/global are the layout
    // `repack_for_marlin_fp4` produced for these dims.
    unsafe {
        ffi::marlin_fp4_gemm_cuda(
            x_ptr as *const Half,
            packed_ptr as *const u32,
            scales_ptr as *const u8,
            global_ptr as *const u16,
            out_ptr as *mut Half,
            c_tmp_ptr as *mut f32,
            ws_ptr as *mut i32,
            i32::try_from(m)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(group_size)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("marlin_fp4_gemm_cuda failed at [m,n,k]=[{m},{n},{k}]: {e}"))
    }
}

/// Widen Marlin NVFP4 tiles + their S0E5M3 scale tail to the dense E4M3 `[n, k]`
/// bytes DeepGEMM's dense NT entry takes as B, with the per-128x128-block power
/// of two divided out (`block_pow2` — DeepGEMM takes it back as `sfb`).
/// `inv_lift` undoes the per-tensor power of two the repack multiplied into the
/// stored scale byte.
///
/// # Safety
/// `output` must cover `n * k` E4M3 bytes on `ctx`'s stream and stay live
/// through the launch (it is the caller's reusable scratch).
#[allow(clippy::too_many_arguments)]
pub unsafe fn dequantize_fp4_marlin_to_fp8(
    ctx: &DeviceContext,
    packed: &impl DevicePtr<u8>,
    global: &impl DevicePtr<f32>,
    block_pow2: &impl DevicePtr<f32>,
    inv_lift: f32,
    output: RawDevicePtr<u8>,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<()> {
    let nk = extent(n, k, "dequantize_fp4_marlin_to_fp8 weight")?;
    ensure!(
        group_size > 0
            && packed.len() >= nk / 2 + nk / group_size
            && global.len() >= 1
            && block_pow2.len() >= (n.div_ceil(128) + 1) * k.div_ceil(128),
        "dequantize_fp4_marlin_to_fp8 buffers do not cover [n,k,gs]=[{n},{k},{group_size}]: packed={} global={} block_pow2={}",
        packed.len(),
        global.len(),
        block_pow2.len()
    );
    let (packed_ptr, _gp) = packed.device_ptr(&ctx.stream);
    let (global_ptr, _gg) = global.device_ptr(&ctx.stream);
    let (pow2_ptr, _gb) = block_pow2.device_ptr(&ctx.stream);
    // SAFETY: source lengths checked above; the output scratch is the caller's
    // contract.
    unsafe {
        ffi::dequantize_fp4_marlin_to_fp8_cuda(
            packed_ptr as *const u8,
            global_ptr as *const f32,
            pow2_ptr as *const f32,
            inv_lift,
            output.as_mut_ptr(),
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(group_size)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dequantize_fp4_marlin_to_fp8_cuda failed at [n,k]=[{n},{k}]: {e}"))
    }
}

/// Widen Marlin NVFP4 tiles + their S0E5M3 scale tail to dense BF16 `[n, k]`.
/// `inv_lift` undoes the per-tensor power of two the repack multiplied into the
/// stored scale byte. Unlike the FP8 twin there is no per-block divisor: this
/// output is the weight itself, not DeepGEMM's B.
///
/// The only lane that reaches a Marlin-repacked NVFP4 weight's values on the
/// host side -- the repack releases the group bytes, so a LoRA merge has
/// nothing else to promote from.
pub fn dequantize_fp4_marlin_to_bf16(
    ctx: &DeviceContext,
    packed: &impl DevicePtr<u8>,
    global: &impl DevicePtr<f32>,
    inv_lift: f32,
    output: &mut impl DevicePtrMut<bf16>,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<()> {
    let nk = extent(n, k, "dequantize_fp4_marlin_to_bf16 weight")?;
    ensure!(
        group_size > 0
            && packed.len() >= nk / 2 + nk / group_size
            && global.len() >= 1
            && output.len() >= nk,
        "dequantize_fp4_marlin_to_bf16 buffers do not cover [n,k,gs]=[{n},{k},{group_size}]: packed={} global={} output={}",
        packed.len(),
        global.len(),
        output.len()
    );
    let (packed_ptr, _gp) = packed.device_ptr(&ctx.stream);
    let (global_ptr, _gg) = global.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; `packed` is the layout
    // `repack_for_marlin_fp4` produced for these dims.
    unsafe {
        ffi::dequantize_fp4_marlin_to_bf16_cuda(
            packed_ptr as *const u8,
            global_ptr as *const f32,
            inv_lift,
            out_ptr as *mut Half,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(group_size)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dequantize_fp4_marlin_to_bf16_cuda failed at [n,k]=[{n},{k}]: {e}"))
    }
}

/// W8A16 Marlin GEMM: C[m,n] = X[m,k] @ dequant(Marlin-packed W). Scratch/// W8A16 Marlin GEMM: C[m,n] = X[m,k] @ dequant(Marlin-packed W). Scratch
/// contract matches `marlin_fp8_gemm`.
#[allow(clippy::too_many_arguments)]
pub fn marlin_w8a16_gemm(
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
    group_size: usize,
) -> Result<()> {
    ensure!(
        input.len() >= extent(m, k, "marlin_w8a16_gemm input")?
            && output.len() >= extent(m, n, "marlin_w8a16_gemm output")?,
        "marlin_w8a16_gemm buffers do not cover [m,n,k]=[{m},{n},{k}]: input={} output={}",
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
    // `repack_for_marlin_w8a16` produced for these dims.
    unsafe {
        ffi::marlin_w8a16_gemm_cuda(
            x_ptr as *const Half,
            packed_ptr as *const u32,
            scales_ptr as *const Half,
            out_ptr as *mut Half,
            c_tmp_ptr as *mut f32,
            ws_ptr as *mut i32,
            i32::try_from(m)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(group_size)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("marlin_w8a16_gemm_cuda failed at [m,n,k]=[{m},{n},{k}]: {e}"))
    }
}

/// W8A16 group-scale grid length: `scales[row * (k/group_size) + col_group]`.
fn int_group_scale_len(n: usize, k: usize, group_size: usize, what: &'static str) -> Result<usize> {
    ensure!(
        group_size > 0 && k.is_multiple_of(group_size),
        "{what} cols {k} not group-aligned to {group_size}"
    );
    extent(n, k / group_size, what)
}

/// Scalar W8A16 GEMV, single row: out[n] = dequant(W[n,k]) @ x[k].
pub fn w8a16_gemv(
    ctx: &DeviceContext,
    weight: &impl DevicePtr<i8>,
    scales: &impl DevicePtr<bf16>,
    input: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<()> {
    ensure!(
        weight.len() >= extent(n, k, "w8a16_gemv weight")?
            && scales.len() >= int_group_scale_len(n, k, group_size, "w8a16_gemv")?
            && input.len() >= k
            && output.len() >= n,
        "w8a16_gemv buffers do not cover [n,k,gs]=[{n},{k},{group_size}]: weight={} scales={} input={} output={}",
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
        ffi::w8a16_gemv_cuda(
            qw_ptr as *const i8,
            scales_ptr as *const Half,
            x_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(group_size)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("w8a16_gemv_cuda failed at [n,k]=[{n},{k}]: {e}"))
    }
}

/// Batched W8A16 GEMV: out[m,n] = X[m,k] @ dequant(W[n,k])^T.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch(
    ctx: &DeviceContext,
    weight: &impl DevicePtr<i8>,
    scales: &impl DevicePtr<bf16>,
    input: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<()> {
    ensure!(
        weight.len() >= extent(n, k, "w8a16_gemv_batch weight")?
            && scales.len() >= int_group_scale_len(n, k, group_size, "w8a16_gemv_batch")?
            && input.len() >= extent(m, k, "w8a16_gemv_batch input")?
            && output.len() >= extent(m, n, "w8a16_gemv_batch output")?,
        "w8a16_gemv_batch buffers do not cover [m,n,k,gs]=[{m},{n},{k},{group_size}]: weight={} scales={} input={} output={}",
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
        ffi::w8a16_gemv_batch_cuda(
            qw_ptr as *const i8,
            scales_ptr as *const Half,
            x_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(m)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(group_size)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("w8a16_gemv_batch_cuda failed at [m,n,k]=[{m},{n},{k}]: {e}"))
    }
}

/// Scalar W4A16 GEMV, single row: out[n] = dequant(W[n,k]) @ x[k]. The weight
/// holds two INT4 nibbles per byte (`n * k / 2`).
pub fn w4a16_gemv(
    ctx: &DeviceContext,
    weight: &impl DevicePtr<i8>,
    scales: &impl DevicePtr<bf16>,
    input: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<()> {
    ensure!(
        weight.len() >= extent(n, k, "w4a16_gemv weight")? / 2
            && scales.len() >= int_group_scale_len(n, k, group_size, "w4a16_gemv")?
            && input.len() >= k
            && output.len() >= n,
        "w4a16_gemv buffers do not cover [n,k,gs]=[{n},{k},{group_size}]: weight={} scales={} input={} output={}",
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
        ffi::w4a16_gemv_cuda(
            qw_ptr as *const u8,
            scales_ptr as *const Half,
            x_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(group_size)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("w4a16_gemv_cuda failed at [n,k]=[{n},{k}]: {e}"))
    }
}

/// Batched W4A16 GEMV: out[m,n] = X[m,k] @ dequant(W[n,k])^T. Weight packing
/// matches `w4a16_gemv`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_batch(
    ctx: &DeviceContext,
    weight: &impl DevicePtr<i8>,
    scales: &impl DevicePtr<bf16>,
    input: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<()> {
    ensure!(
        weight.len() >= extent(n, k, "w4a16_gemv_batch weight")? / 2
            && scales.len() >= int_group_scale_len(n, k, group_size, "w4a16_gemv_batch")?
            && input.len() >= extent(m, k, "w4a16_gemv_batch input")?
            && output.len() >= extent(m, n, "w4a16_gemv_batch output")?,
        "w4a16_gemv_batch buffers do not cover [m,n,k,gs]=[{m},{n},{k},{group_size}]: weight={} scales={} input={} output={}",
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
        ffi::w4a16_gemv_batch_cuda(
            qw_ptr as *const u8,
            scales_ptr as *const Half,
            x_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(m)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(group_size)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("w4a16_gemv_batch_cuda failed at [m,n,k]=[{m},{n},{k}]: {e}"))
    }
}

/// Dequant an INT8 group-scaled W8A16 weight `[n, k]` into a resident BF16
/// buffer.
///
/// # Safety
/// `output` must cover `n * k` bf16 elements on `ctx`'s stream and stay live
/// through the launch (it is the caller's reusable scratch).
pub unsafe fn dequantize_w8a16_to_bf16(
    ctx: &DeviceContext,
    weight: &impl DevicePtr<i8>,
    scales: &impl DevicePtr<bf16>,
    output: RawDevicePtr<bf16>,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<()> {
    ensure!(
        weight.len() >= extent(n, k, "dequantize_w8a16_to_bf16 weight")?
            && scales.len() >= int_group_scale_len(n, k, group_size, "dequantize_w8a16_to_bf16")?,
        "dequantize_w8a16_to_bf16 buffers do not cover [n,k,gs]=[{n},{k},{group_size}]: weight={} scales={}",
        weight.len(),
        scales.len()
    );
    let (qw_ptr, _gqw) = weight.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    // SAFETY: source lengths checked above; the output scratch is the caller's
    // contract.
    unsafe {
        ffi::dequantize_w8a16_to_bf16_cuda(
            qw_ptr as *const i8,
            scales_ptr as *const Half,
            output.as_mut_ptr().cast(),
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(group_size)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("dequantize_w8a16_to_bf16_cuda failed at [n,k]=[{n},{k}]: {e}"))
    }
}

/// Marlin `c_tmp` scratch size in f32 elements for a max row count `m` on
/// `sms` SMs. Pure host-side arithmetic, no device work.
pub fn marlin_c_tmp_floats(m: usize, sms: usize) -> Result<usize> {
    // SAFETY: pure size query (arithmetic on m/sms), no device work.
    Ok(unsafe { ffi::marlin_c_tmp_floats(i32::try_from(m)?, i32::try_from(sms)?) } as usize)
}

/// Marlin lock workspace size in i32 elements for `sms` SMs. Pure host-side
/// arithmetic, no device work.
pub fn marlin_workspace_ints(sms: usize) -> Result<usize> {
    // SAFETY: pure size query (arithmetic on sms), no device work.
    Ok(unsafe { ffi::marlin_workspace_ints(i32::try_from(sms)?) } as usize)
}

/// Quantize a dense BF16 weight `[n, k]` into FP8 E4M3 bytes + per-block f32
/// scales (amax/448). The scale grid is `[ceil(n/block_m), ceil(k/block_k)]`.
#[allow(clippy::too_many_arguments)]
pub fn quantize_bf16_to_fp8_block_scaled(
    ctx: &DeviceContext,
    input: &impl DevicePtr<bf16>,
    weight: &mut impl DevicePtrMut<u8>,
    scales: &mut impl DevicePtrMut<f32>,
    n: usize,
    k: usize,
    block_m: usize,
    block_k: usize,
) -> Result<()> {
    let nk = extent(n, k, "quantize_bf16_to_fp8_block_scaled weight")?;
    ensure!(
        block_m > 0
            && block_k > 0
            && input.len() >= nk
            && weight.len() >= nk
            && scales.len()
                >= extent(
                    n.div_ceil(block_m),
                    k.div_ceil(block_k),
                    "quantize_bf16_to_fp8_block_scaled scales"
                )?,
        "quantize_bf16_to_fp8_block_scaled buffers do not cover [n,k,bm,bk]=[{n},{k},{block_m},{block_k}]: input={} weight={} scales={}",
        input.len(),
        weight.len(),
        scales.len()
    );
    let (in_ptr, _gi) = input.device_ptr(&ctx.stream);
    let (qw_ptr, _gq) = weight.device_ptr_mut(&ctx.stream);
    let (sc_ptr, _gs) = scales.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths checked above; all buffers belong to `ctx.stream`.
    unsafe {
        ffi::quantize_bf16_to_fp8_block_scaled_cuda(
            in_ptr as *const Half,
            qw_ptr as *mut u8,
            sc_ptr as *mut f32,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(block_m)?,
            i32::try_from(block_k)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| {
            anyhow!("quantize_bf16_to_fp8_block_scaled_cuda failed at [n,k]=[{n},{k}]: {e}")
        })
    }
}

/// One DSv4 block-scaled batch-GEMV symbol (`dsv4_fp{8,4}_gemv_batch_cuda`
/// share this signature).
type Dsv4GemvBatchFn = unsafe extern "C" fn(
    *const u8,
    *const u8,
    *const Half,
    *mut Half,
    i32,
    i32,
    i32,
    i32,
    i32,
    CUstream,
) -> CUresult;

#[allow(clippy::too_many_arguments)]
fn dsv4_gemv_batch(
    ctx: &DeviceContext,
    symbol: &'static str,
    kernel: Dsv4GemvBatchFn,
    weight_len_elems: usize,
    weight: &impl DevicePtr<i8>,
    scales: &impl DevicePtr<u8>,
    input: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
    m: usize,
    n: usize,
    k: usize,
    scale_rows: usize,
    scale_cols: usize,
) -> Result<()> {
    ensure!(
        weight.len() >= weight_len_elems
            && scales.len() >= extent(scale_rows, scale_cols, "dsv4_gemv_batch scales")?
            && input.len() >= extent(m, k, "dsv4_gemv_batch input")?
            && output.len() >= extent(m, n, "dsv4_gemv_batch output")?,
        "{symbol} buffers do not cover [m,n,k]=[{m},{n},{k}]: weight={} scales={} input={} output={}",
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
        kernel(
            qw_ptr as *const u8,
            scales_ptr as *const u8,
            x_ptr as *const Half,
            out_ptr as *mut Half,
            i32::try_from(m)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(scale_rows)?,
            i32::try_from(scale_cols)?,
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("{symbol} failed at [m,n,k]=[{m},{n},{k}]: {e}"))
    }
}

/// Batched DSv4 FP8 block-scaled GEMV: out[m,n] = X[m,k] @ dequant(W[n,k])^T.
/// `scales` is the DSv4 E8M0 byte grid `[scale_rows, scale_cols]`.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_fp8_gemv_batch(
    ctx: &DeviceContext,
    weight: &impl DevicePtr<i8>,
    scales: &impl DevicePtr<u8>,
    input: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
    m: usize,
    n: usize,
    k: usize,
    scale_rows: usize,
    scale_cols: usize,
) -> Result<()> {
    let nk = extent(n, k, "dsv4_fp8_gemv_batch weight")?;
    dsv4_gemv_batch(
        ctx,
        "dsv4_fp8_gemv_batch_cuda",
        ffi::dsv4_fp8_gemv_batch_cuda,
        nk,
        weight,
        scales,
        input,
        output,
        m,
        n,
        k,
        scale_rows,
        scale_cols,
    )
}

/// Batched DSv4 FP4 block-scaled GEMV (weight holds two E2M1 nibbles per
/// byte, `n*k/2`). Scale layout matches [`dsv4_fp8_gemv_batch`].
#[allow(clippy::too_many_arguments)]
pub fn dsv4_fp4_gemv_batch(
    ctx: &DeviceContext,
    weight: &impl DevicePtr<i8>,
    scales: &impl DevicePtr<u8>,
    input: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
    m: usize,
    n: usize,
    k: usize,
    scale_rows: usize,
    scale_cols: usize,
) -> Result<()> {
    let nk = extent(n, k, "dsv4_fp4_gemv_batch weight")?;
    dsv4_gemv_batch(
        ctx,
        "dsv4_fp4_gemv_batch_cuda",
        ffi::dsv4_fp4_gemv_batch_cuda,
        nk / 2,
        weight,
        scales,
        input,
        output,
        m,
        n,
        k,
        scale_rows,
        scale_cols,
    )
}

/// One DSv4 routed batch-GEMV symbol (`dsv4_fp{8,4}_route_gemv_batch_cuda`
/// share this signature).
type Dsv4RouteGemvBatchFn = unsafe extern "C" fn(
    *const u64,
    *const u64,
    *const Half,
    *mut Half,
    *const i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    CUstream,
) -> CUresult;

/// Shared launch arguments of the two DSv4 routed batch-GEMV variants: each
/// route multiplies one `[n, k]` expert (from the per-expert device pointer
/// tables) against its input row. `route_meta: None` selects expert
/// `route % experts_per_rank` over token-major identity routing, which is the
/// layout the extent checks below assume.
pub struct Dsv4RouteGemvArgs<'a> {
    pub route_meta: Option<&'a CudaSlice<i32>>,
    pub local_expert_start: usize,
    pub experts_per_rank: usize,
    pub num_routes: usize,
    pub n: usize,
    pub k: usize,
    pub scale_rows: usize,
    pub scale_cols: usize,
    pub apply_route_weight: bool,
}

fn dsv4_route_gemv_batch(
    ctx: &DeviceContext,
    symbol: &'static str,
    kernel: Dsv4RouteGemvBatchFn,
    weight_ptrs: &impl DevicePtr<u64>,
    scale_ptrs: &impl DevicePtr<u64>,
    input: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
    args: Dsv4RouteGemvArgs<'_>,
) -> Result<()> {
    let Dsv4RouteGemvArgs {
        route_meta,
        local_expert_start,
        experts_per_rank,
        num_routes,
        n,
        k,
        scale_rows,
        scale_cols,
        apply_route_weight,
    } = args;
    ensure!(
        weight_ptrs.len() >= experts_per_rank
            && scale_ptrs.len() >= experts_per_rank
            && route_meta.is_none_or(|meta| meta.len() >= num_routes)
            && input.len() >= extent(num_routes, k, "dsv4_route_gemv_batch input")?
            && output.len() >= extent(num_routes, n, "dsv4_route_gemv_batch output")?,
        "{symbol} buffers do not cover [routes,n,k,experts]=[{num_routes},{n},{k},{experts_per_rank}]: weight_ptrs={} scale_ptrs={} input={} output={}",
        weight_ptrs.len(),
        scale_ptrs.len(),
        input.len(),
        output.len()
    );
    let (wp_ptr, _gw) = weight_ptrs.device_ptr(&ctx.stream);
    let (sp_ptr, _gs) = scale_ptrs.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = input.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    let meta_guard = route_meta.map(|meta| meta.device_ptr(&ctx.stream));
    let meta_ptr = meta_guard
        .as_ref()
        .map_or(std::ptr::null(), |(ptr, _g)| *ptr as *const i32);
    // SAFETY: lengths checked above; the pointer tables hold `experts_per_rank`
    // live per-expert weight/scale device addresses (the caller's load-time
    // contract).
    unsafe {
        kernel(
            wp_ptr as *const u64,
            sp_ptr as *const u64,
            x_ptr as *const Half,
            out_ptr as *mut Half,
            meta_ptr,
            i32::try_from(local_expert_start)?,
            i32::try_from(experts_per_rank)?,
            i32::try_from(num_routes)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(scale_rows)?,
            i32::try_from(scale_cols)?,
            i32::from(apply_route_weight),
            ctx.stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("{symbol} failed at [routes,n,k]=[{num_routes},{n},{k}]: {e}"))
    }
}

pub fn dsv4_fp8_route_gemv_batch(
    ctx: &DeviceContext,
    weight_ptrs: &impl DevicePtr<u64>,
    scale_ptrs: &impl DevicePtr<u64>,
    input: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
    args: Dsv4RouteGemvArgs<'_>,
) -> Result<()> {
    dsv4_route_gemv_batch(
        ctx,
        "dsv4_fp8_route_gemv_batch_cuda",
        ffi::dsv4_fp8_route_gemv_batch_cuda,
        weight_ptrs,
        scale_ptrs,
        input,
        output,
        args,
    )
}

pub fn dsv4_fp4_route_gemv_batch(
    ctx: &DeviceContext,
    weight_ptrs: &impl DevicePtr<u64>,
    scale_ptrs: &impl DevicePtr<u64>,
    input: &impl DevicePtr<bf16>,
    output: &mut impl DevicePtrMut<bf16>,
    args: Dsv4RouteGemvArgs<'_>,
) -> Result<()> {
    dsv4_route_gemv_batch(
        ctx,
        "dsv4_fp4_route_gemv_batch_cuda",
        ffi::dsv4_fp4_route_gemv_batch_cuda,
        weight_ptrs,
        scale_ptrs,
        input,
        output,
        args,
    )
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
