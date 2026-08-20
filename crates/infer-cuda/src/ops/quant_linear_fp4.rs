use anyhow::{Result, anyhow, bail};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix};
use cuda_kernels::tensor::{RawDevicePtr, WeightFormat, cache_ptr};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use std::sync::atomic::{AtomicU64, Ordering};

use super::QWEN_FP4_DEEPGEMM_MIN_M;
use super::fp8::{
    deepgemm_dense_nt, dense_deepgemm_prefill_floor, qwen_fp8_deepgemm_dense_enabled,
    qwen_fp8_dense_sm_supports_deepgemm, with_e4m3_weight_scratch,
};
use super::{marlin_sm_supported, qwen_quant_profile, with_marlin_scratch};

pub(super) static MARLIN_FP4_HITS: AtomicU64 = AtomicU64::new(0);
pub(super) static FP4_DEEPGEMM_HITS: AtomicU64 = AtomicU64::new(0);

/// NVFP4 has exactly one serving path on every SM tier that can run it: Marlin
/// at decode, and the same layout widened to E4M3 for DeepGEMM above
/// [`dense_deepgemm_prefill_floor`] at prefill. The dequant->BF16->cuBLAS and
/// scalar-GEMV arms this used to fall back to are gone: they were reachable
/// only on sm_70 (where nothing serves NVFP4 — the V100 lane is W4A16, a
/// different format with its own dispatch) or on a shape `repack_for_marlin_fp4`
/// declined, which now fails at load with the reason instead of degrading
/// silently to a path no gate has ever executed.
fn fp4_marlin_ready(ctx: &DeviceContext, weight: &DeviceMatrix) -> bool {
    weight.marlin_packed.is_some() && weight.marlin_scales.is_some() && marlin_sm_supported(ctx)
}

/// Whether the NVFP4 DeepGEMM prefill arm can serve this weight at some M. The
/// loader asks before paying for the `sfb` the arm needs; the arm itself asks
/// again per call.
pub(crate) fn fp4_deepgemm_available(ctx: &DeviceContext, weight: &DeviceMatrix) -> bool {
    weight.weight_format == WeightFormat::Fp4E2M1Group
        && weight.group_size == 16
        && weight.rows.is_multiple_of(64)
        && weight.cols.is_multiple_of(128)
        && qwen_fp8_dense_sm_supports_deepgemm(ctx)
        && qwen_fp8_deepgemm_dense_enabled()
        && dense_deepgemm_prefill_floor(QWEN_FP4_DEEPGEMM_MIN_M).is_some()
}

/// Widen NVFP4 to E4M3 into scratch, then contract it on the FP8 tensor cores.
///
/// The widen reads the Marlin layout, not the checkpoint nibbles — same byte
/// volume, and the checkpoint copy is freed at load. It therefore inherits the
/// repack's one lossy step: a group scale far enough below the tensor max
/// flushes to zero (`repack_for_marlin_fp4`), which makes prefill agree with
/// the Marlin decode arm rather than with the checkpoint.
///
/// The weight's `fp4_deepgemm_sfb` carries the per-128x128-block power of two
/// the widening divided out, so DeepGEMM multiplies it back into the fp32
/// accumulator and the result is the same product Marlin forms — up to the one
/// mantissa bit E4M3 cannot hold (`prepare_fp4_deepgemm_sfb`) and the E4M3
/// activation rounding this entry point forces, neither of which is free. That
/// is why the arm is gated on the needle ladder and not on the scale algebra.

/// Compile the NVFP4 prefill arm's DeepGEMM kernel at load, and reserve the
/// E4M3 weight scratch while doing it.
///
/// Without this the arm relies on a coincidence: this checkpoint's NVFP4 and
/// per-channel FP8 MLP weights share `[34816, 5120]` and `[5120, 17408]`, so the
/// FP8 warm happens to build a cubin the FP4 arm can reuse. A checkpoint whose
/// shapes do not collide would JIT inside the first request instead.
pub(crate) fn warm_fp4_deepgemm_dense(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    seq_len: usize,
) -> Result<bool> {
    if weight.weight_format != WeightFormat::Fp4E2M1Group
        || weight.fp4_deepgemm_sfb.is_none()
        || !fp4_deepgemm_available(ctx, weight)
    {
        return Ok(false);
    }
    let Some(floor) = dense_deepgemm_prefill_floor(QWEN_FP4_DEEPGEMM_MIN_M) else {
        return Ok(false);
    };
    let m = seq_len.max(floor);
    let n = weight.rows;
    let k = weight.cols;
    let input = ctx
        .stream
        .alloc_zeros::<bf16>(m * k)
        .map_err(|e| anyhow!("NVFP4 DeepGEMM warm input alloc failed: {e}"))?;
    let mut out = ctx
        .stream
        .alloc_zeros::<bf16>(m * n)
        .map_err(|e| anyhow!("NVFP4 DeepGEMM warm output alloc failed: {e}"))?;
    // The real arm, on throwaway buffers: same widen, same launch config, so it
    // builds the kernel the first request will replay.
    try_fp4_deepgemm_gemm(ctx, weight, &input, &mut out, m)
}
fn try_fp4_deepgemm_gemm(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<bool> {
    if weight.weight_format != WeightFormat::Fp4E2M1Group || weight.fp4_deepgemm_sfb.is_none() {
        return Ok(false);
    }
    let Some(floor) = dense_deepgemm_prefill_floor(QWEN_FP4_DEEPGEMM_MIN_M) else {
        return Ok(false);
    };
    if m < floor || !fp4_deepgemm_available(ctx, weight) {
        return Ok(false);
    }
    let packed = weight
        .marlin_packed
        .as_ref()
        .ok_or_else(|| anyhow!("fp4_e2m1_group missing marlin_packed"))?;
    let global = weight
        .scale_f32
        .as_ref()
        .ok_or_else(|| anyhow!("fp4_e2m1_group missing scale_f32 global scale"))?;
    let sfb = weight.fp4_deepgemm_sfb.as_ref().expect("checked above");
    let n = weight.rows;
    let k = weight.cols;
    // The E4M3 copy lives in the shared FP8 scratch; the GEMM is ordered after
    // the widen on the same stream.
    let Some(b) = with_e4m3_weight_scratch(ctx, n * k, |dst| -> Result<RawDevicePtr<u8>> {
        let dst_ptr = cache_ptr(dst, ctx);
        qwen_quant_profile(ctx, "qwen/fp4/dense_widen_fp8", m, n, k, || {
            // SAFETY: ptrs from live device allocations sized to the dims passed.
            unsafe {
                ffi::dequantize_fp4_marlin_to_fp8_cuda(
                    cache_ptr(packed, ctx).as_ptr(),
                    cache_ptr(global, ctx).as_ptr(),
                    cache_ptr(sfb, ctx).as_ptr(),
                    weight.fp4_marlin_scale_lift_inv,
                    dst_ptr.as_mut_ptr(),
                    n as i32,
                    k as i32,
                    weight.group_size as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("NVFP4 widen-to-E4M3 kernel failed: {e}"))
            }
        })?;
        Ok(dst_ptr)
    })?
    else {
        return Ok(false);
    };
    deepgemm_dense_nt(ctx, input, output, b, cache_ptr(sfb, ctx), m, n, k, "NVFP4")?;
    Ok(true)
}

/// The NVFP4 Marlin launch. `fp4_marlin_ready` is the caller's gate; a
/// repacked weight's pre-repack bytes are freed at load, so this is the only
/// consumer once the DeepGEMM prefill arm has declined.
fn marlin_fp4_gemm_raw(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<()> {
    let (Some(packed), Some(global)) =
        (weight.marlin_packed.as_ref(), weight.marlin_scales.as_ref())
    else {
        unreachable!("fp4_marlin_ready returns true only when both are Some")
    };
    let n = weight.rows; // output dim
    let k = weight.cols; // contraction
    let (packed_ptr, _gp) = packed.device_ptr(&ctx.stream);
    // The S0E5M3 group scales sit in the tail of the same allocation; see
    // `repack_for_marlin_fp4`.
    let scales_ptr = packed_ptr + (n * k / 2) as u64;
    let (global_ptr, _gg) = global.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = input.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();
    with_marlin_scratch(ctx, |scratch| {
        let c_tmp = scratch.c_tmp.as_ref().unwrap();
        let workspace = scratch.workspace.as_ref().unwrap();
        let (c_tmp_ptr, _gc) = c_tmp.device_ptr(&ctx.stream);
        let (ws_ptr, _gw) = workspace.device_ptr(&ctx.stream);
        qwen_quant_profile(ctx, "qwen/fp4/marlin_gemm", m, n, k, || {
            // SAFETY: all ptrs from live device allocations; packed/scales/global
            // sized by repack_for_marlin_fp4 for these dims, x=[m,k], out=[m,n],
            // c_tmp/workspace sized to the SM max.
            unsafe {
                ffi::marlin_fp4_gemm_cuda(
                    x_ptr as *const ffi::Half,
                    packed_ptr as *const u32,
                    scales_ptr as *const u8,
                    global_ptr as *const u16,
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
                .map_err(|e| anyhow!("NVFP4 Marlin GEMM failed: {e}"))
            }
        })?;
        Ok(())
    })?;
    MARLIN_FP4_HITS.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Fp4Route {
    DeepGemm,
    Marlin,
}

#[derive(Clone, Copy)]
pub(super) struct Fp4Query {
    pub(super) marlin_ready: bool,
    pub(super) sfb: bool,
    pub(super) prefill_shape: bool,
}

/// The static part of the FP4 order. The DeepGEMM arm keeps the dynamic floor
/// (`dense_deepgemm_prefill_floor`) and its own shape/SM gates; this fn only
/// says which arm owns the M.
pub(super) fn fp4_route(q: Fp4Query, m: usize) -> Fp4Route {
    if q.sfb && q.prefill_shape && m >= QWEN_FP4_DEEPGEMM_MIN_M {
        Fp4Route::DeepGemm
    } else {
        Fp4Route::Marlin
    }
}

pub(super) fn run(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<()> {
    let query = Fp4Query {
        marlin_ready: fp4_marlin_ready(ctx, weight),
        sfb: weight.fp4_deepgemm_sfb.is_some(),
        prefill_shape: fp4_deepgemm_available(ctx, weight),
    };
    // DeepGEMM ahead of Marlin because Marlin claims every M — the two are
    // separated by the M floor inside the arm, not by `fp4_marlin_ready`.
    if fp4_route(query, m) == Fp4Route::DeepGemm
        && try_fp4_deepgemm_gemm(ctx, weight, input, output, m)?
    {
        FP4_DEEPGEMM_HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    }
    if query.marlin_ready {
        marlin_fp4_gemm_raw(ctx, weight, input, output, m)?;
        return Ok(());
    }
    bail!("fp4_e2m1_group has no consumable representation")
}
