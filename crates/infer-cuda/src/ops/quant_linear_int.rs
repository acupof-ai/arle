use anyhow::{Result, anyhow, ensure};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix};
use cuda_kernels::tensor::WeightFormat;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use std::sync::atomic::{AtomicU64, Ordering};

use super::QWEN_FP8_DEQUANT_GEMM_MIN_M;
use super::fp8::with_dequant_weight_scratch;
use super::{marlin_sm_supported, qwen_quant_profile, with_marlin_scratch};

pub(super) static W8A16_DEQUANT_GEMM_HITS: AtomicU64 = AtomicU64::new(0);
pub(super) static MARLIN_W8A16_HITS: AtomicU64 = AtomicU64::new(0);
pub(super) static W8A16_GEMV_HITS: AtomicU64 = AtomicU64::new(0);
pub(super) static W4A16_GEMV_HITS: AtomicU64 = AtomicU64::new(0);

/// W8A16 Marlin tensor-core GEMM (Ampere+): C[m,n] = X[m,k] @ dequant(W). Fires
/// when the SM gate is on AND the weight was Marlin-repacked at load
/// (`marlin_packed`/`marlin_scales` present, set by `repack_for_marlin_w8a16`).
/// Supersedes both the scalar batched-GEMV (decode) and the dequant→cuBLAS
/// fallback (prefill) — the win is tensor cores in decode, ~bf16 speed at INT8
/// VRAM. Returns false (→ existing fallbacks) when off or not prepacked.
fn marlin_w8a16_gemm_raw(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<bool> {
    if weight.weight_format != WeightFormat::W8A16 || !marlin_sm_supported(ctx) {
        return Ok(false);
    }
    let (Some(packed), Some(scales)) =
        (weight.marlin_packed.as_ref(), weight.marlin_scales.as_ref())
    else {
        return Ok(false); // not repacked (unaligned shape) → fallback
    };
    let n = weight.rows; // output dim
    let k = weight.cols; // contraction
    let (packed_ptr, _gp) = packed.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = input.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();
    with_marlin_scratch(ctx, |scratch| {
        let c_tmp = scratch.c_tmp.as_ref().unwrap();
        let workspace = scratch.workspace.as_ref().unwrap();
        let (c_tmp_ptr, _gc) = c_tmp.device_ptr(&ctx.stream);
        let (ws_ptr, _gw) = workspace.device_ptr(&ctx.stream);
        qwen_quant_profile(ctx, "qwen/w8a16/marlin_gemm", m, n, k, || {
            // SAFETY: all ptrs from live device allocations; packed/scales sized by
            // repack_for_marlin_w8a16 for these dims, x=[m,k], out=[m,n],
            // c_tmp/workspace sized to the SM max above.
            unsafe {
                ffi::marlin_w8a16_gemm_cuda(
                    x_ptr as *const ffi::Half,
                    packed_ptr as *const u32,
                    scales_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    c_tmp_ptr as *mut f32,
                    ws_ptr as *mut i32,
                    m as i32,
                    n as i32,
                    k as i32,
                    weight.group_size as i32,
                    stream,
                )
                .result()
                .map_err(|e| anyhow!("W8A16 Marlin GEMM failed: {e}"))
            }
        })?;
        Ok(())
    })?;
    MARLIN_W8A16_HITS.fetch_add(1, Ordering::Relaxed);
    Ok(true)
}

/// W8A16 large-M (prefill) path: dequant INT8→BF16 once, then one cuBLAS BF16
/// GEMM over all M rows. Mirrors the FP8 dequant arm — the scalar
/// GEMV re-reads the weight per token (~20× slower at M=2048 and the cause of
/// W8A16's 6× TTFT), so prefill must dequant once and GEMM instead. Small-M
/// decode (`M < QWEN_FP8_DEQUANT_GEMM_MIN_M`) keeps the batched GEMV.
fn try_w8a16_dequant_bf16_gemm(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<bool> {
    ensure!(
        weight.group_size > 0 && weight.cols.is_multiple_of(weight.group_size),
        "W8A16 cols {} not group-aligned to {}",
        weight.cols,
        weight.group_size
    );
    let qw = weight
        .qweight
        .as_ref()
        .ok_or_else(|| anyhow!("W8A16 missing qweight"))?;
    let scales = weight
        .qscales
        .as_ref()
        .ok_or_else(|| anyhow!("W8A16 missing qscales"))?;
    let n = weight.rows; // GEMM M dim (weight rows)
    let k = weight.cols; // GEMM K dim (contraction)
    let weight_elems = n * k;

    // Reuse the FP8 dequant scratch — it is just a format-agnostic BF16 weight
    // buffer sized to the largest weight seen this thread.
    let Some(()) = with_dequant_weight_scratch(ctx, weight_elems, |weight_bf16| -> Result<()> {
        let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
        let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
        let (wbf16_ptr, _gw) = weight_bf16.device_ptr(&ctx.stream);
        let stream = ctx.stream.cu_stream();
        qwen_quant_profile(
            ctx,
            "qwen/w8a16/dense_dequant_bf16",
            m,
            n,
            k,
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            || unsafe {
                ffi::dequantize_w8a16_to_bf16_cuda(
                    qw_ptr as *const i8,
                    scales_ptr as *const ffi::Half,
                    wbf16_ptr as *mut ffi::Half,
                    n as i32,
                    k as i32,
                    weight.group_size as i32,
                    stream,
                )
                .result()
                .map_err(|e| anyhow!("Qwen W8A16 dense dequant kernel failed: {e}"))
            },
        )?;
        let (x_ptr, _gx) = input.device_ptr(&ctx.stream);
        let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
        qwen_quant_profile(
            ctx,
            "qwen/w8a16/dense_dequant_gemm",
            m,
            n,
            k,
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            || unsafe {
                ffi::gemm_cuda(
                    wbf16_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    n as i32,
                    m as i32,
                    k as i32,
                    stream,
                )
                .result()
                .map_err(|e| anyhow!("Qwen W8A16 dense dequant BF16 GEMM failed: {e}"))
            },
        )?;
        Ok(())
    })?
    else {
        return Ok(false);
    };
    Ok(true)
}

/// W8A16 scalar/batched GEMV — the terminal fallback. M=1 takes the singular
/// ABI the decode lane always used; larger M takes the batched ABI.
fn w8a16_gemv(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<()> {
    let qw = weight
        .qweight
        .as_ref()
        .ok_or_else(|| anyhow!("W8A16 missing qweight"))?;
    let scales = weight
        .qscales
        .as_ref()
        .ok_or_else(|| anyhow!("W8A16 missing qscales"))?;
    ensure!(
        weight.group_size > 0 && weight.cols.is_multiple_of(weight.group_size),
        "W8A16 cols {} not group-aligned to {}",
        weight.cols,
        weight.group_size
    );
    let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = input.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();
    // SAFETY: ptrs from live device allocations sized to the dims passed.
    unsafe {
        if m == 1 {
            ffi::w8a16_gemv_cuda(
                qw_ptr as *const i8,
                scales_ptr as *const ffi::Half,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                weight.rows as i32,
                weight.cols as i32,
                weight.group_size as i32,
                stream,
            )
            .result()?;
        } else {
            ffi::w8a16_gemv_batch_cuda(
                qw_ptr as *const i8,
                scales_ptr as *const ffi::Half,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                m as i32,
                weight.rows as i32,
                weight.cols as i32,
                weight.group_size as i32,
                stream,
            )
            .result()?;
        }
    }
    Ok(())
}

/// W4A16 scalar/batched GEMV — W4A16's only serving path.
fn w4a16_gemv(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<()> {
    let qw = weight
        .qweight
        .as_ref()
        .ok_or_else(|| anyhow!("W4A16 missing qweight"))?;
    let scales = weight
        .qscales
        .as_ref()
        .ok_or_else(|| anyhow!("W4A16 missing qscales"))?;
    ensure!(
        weight.group_size > 0 && weight.cols.is_multiple_of(weight.group_size),
        "W4A16 cols {} not group-aligned to {}",
        weight.cols,
        weight.group_size
    );
    let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = input.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();
    // SAFETY: ptrs from live device allocations sized to the dims passed.
    unsafe {
        if m == 1 {
            ffi::w4a16_gemv_cuda(
                qw_ptr as *const u8,
                scales_ptr as *const ffi::Half,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                weight.rows as i32,
                weight.cols as i32,
                weight.group_size as i32,
                stream,
            )
            .result()?;
        } else {
            ffi::w4a16_gemv_batch_cuda(
                qw_ptr as *const u8,
                scales_ptr as *const ffi::Half,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                m as i32,
                weight.rows as i32,
                weight.cols as i32,
                weight.group_size as i32,
                stream,
            )
            .result()?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum IntRoute {
    Marlin,
    DequantGemm,
    Gemv,
}

#[derive(Clone, Copy)]
pub(super) struct IntQuery {
    pub(super) repacked: bool,
    pub(super) source: bool,
    pub(super) is_w8a16: bool,
}

pub(super) fn int_route(q: IntQuery, m: usize, sm_marlin: bool) -> IntRoute {
    if q.is_w8a16 && q.repacked && sm_marlin {
        return IntRoute::Marlin;
    }
    if q.is_w8a16 && q.source && m >= QWEN_FP8_DEQUANT_GEMM_MIN_M {
        return IntRoute::DequantGemm;
    }
    IntRoute::Gemv
}

pub(super) fn run(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<()> {
    let query = IntQuery {
        repacked: weight.marlin_packed.is_some() && weight.marlin_scales.is_some(),
        source: weight.qweight.is_some() && weight.qscales.is_some(),
        is_w8a16: weight.weight_format == WeightFormat::W8A16,
    };
    match int_route(query, m, marlin_sm_supported(ctx)) {
        IntRoute::Marlin => {
            if !marlin_w8a16_gemm_raw(ctx, weight, input, output, m)? {
                w8a16_gemv(ctx, weight, input, output, m)?;
                W8A16_GEMV_HITS.fetch_add(1, Ordering::Relaxed);
            }
        }
        IntRoute::DequantGemm => {
            if try_w8a16_dequant_bf16_gemm(ctx, weight, input, output, m)? {
                W8A16_DEQUANT_GEMM_HITS.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
            w8a16_gemv(ctx, weight, input, output, m)?;
            W8A16_GEMV_HITS.fetch_add(1, Ordering::Relaxed);
        }
        IntRoute::Gemv => {
            if query.is_w8a16 {
                w8a16_gemv(ctx, weight, input, output, m)?;
                W8A16_GEMV_HITS.fetch_add(1, Ordering::Relaxed);
            } else {
                w4a16_gemv(ctx, weight, input, output, m)?;
                W4A16_GEMV_HITS.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    Ok(())
}
