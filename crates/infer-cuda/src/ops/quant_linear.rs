use anyhow::{Result, anyhow, bail, ensure};
use cuda_kernels::ffi;
use cuda_kernels::moe as cuda_moe;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::tensor::{WeightFormat, cache_ptr};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, sys::CUevent_flags};
use half::bf16;
use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

// Dense-GEMM floor for FP8 weights: at or above this M, DeepGEMM dense (or
// the dequant+cuBLAS fallback) takes the GEMM; below it the scalar
// warp-per-row GEMV keeps decode-sized calls. The old 1024 floor sent every
// sub-1024 prefill chunk to the GEMV the code itself forbids for prefill —
// measured on the agent-OPD toy round: 52k gemv_batch calls, zero DeepGEMM,
// prefill 138 tok/s vs 453 tok/s for the same trajectory's training forward
// (docs/plans/rollout-optimization.md). Tool-result tail prefills are 64-128
// tokens, so the floor belongs at the decode/prefill boundary, not at 1024.
// Lowered 64→16 for the DSpark block-16 spec verify: the same bug class one
// tier down — B=16 on the batched GEMV measured dense_ffn 94 ms/step vs the
// DeepGEMM lane's ~6 ms at the same row count (H20, Qwen3.6-27B-FP8); the
// masked DeepGEMM tile makes M=16 cost ≈ one weight pass. Spec depths ≤8 stay
// on the TILE==B amortizing GEMV (1.04-1.14× a single decode — already optimal).
const QWEN_FP8_DEEPGEMM_DENSE_MIN_M: usize = 16;

#[derive(Default)]
struct QwenFp8DenseScratch {
    input_fp8: Option<CudaSlice<u8>>,
    input_fp8_capacity: usize,
    input_scales: Option<CudaSlice<f32>>,
    input_scales_capacity: usize,
    active_experts: Option<CudaSlice<i32>>,
    active_offsets: Option<CudaSlice<i32>>,
    active_counts: Option<CudaSlice<i32>>,
}

/// Reusable per-thread BF16 weight scratch for the pre-Hopper dense FP8
/// fallback (`dequantize_fp8_block_scaled_to_bf16` → `gemm_cuda`). Sized to the
/// largest dense FP8 projection `[rows, cols]` seen; grows monotonically so the
/// dequant is a single device-resident buffer reused across steps (no per-step
/// alloc churn). Only allocated on sm < 9 — Hopper never touches this.
#[derive(Default)]
struct QwenFp8DequantScratch {
    weight_bf16: Option<CudaSlice<bf16>>,
    capacity: usize,
}

thread_local! {
    static QWEN_FP8_DEQUANT_SCRATCH: RefCell<QwenFp8DequantScratch> =
        RefCell::new(QwenFp8DequantScratch::default());
}

thread_local! {
    static QWEN_FP8_DENSE_SCRATCH: RefCell<QwenFp8DenseScratch> =
        RefCell::new(QwenFp8DenseScratch::default());
}

impl QwenFp8DenseScratch {
    fn ensure(&mut self, ctx: &DeviceContext, input_len: usize, scale_len: usize) -> Result<()> {
        if self.input_fp8_capacity < input_len {
            self.input_fp8 = Some(
                ctx.stream
                    .alloc_zeros::<u8>(input_len)
                    .map_err(|e| anyhow!("Qwen FP8 dense DeepGEMM input alloc failed: {e}"))?,
            );
            self.input_fp8_capacity = input_len;
        }
        if self.input_scales_capacity < scale_len {
            self.input_scales = Some(
                ctx.stream
                    .alloc_zeros::<f32>(scale_len)
                    .map_err(|e| anyhow!("Qwen FP8 dense DeepGEMM scale alloc failed: {e}"))?,
            );
            self.input_scales_capacity = scale_len;
        }
        if self.active_experts.is_none() {
            self.active_experts =
                Some(ctx.stream.clone_htod(&[0i32]).map_err(|e| {
                    anyhow!("Qwen FP8 dense DeepGEMM active_experts H2D failed: {e}")
                })?);
        }
        if self.active_offsets.is_none() {
            self.active_offsets =
                Some(ctx.stream.clone_htod(&[0i32]).map_err(|e| {
                    anyhow!("Qwen FP8 dense DeepGEMM active_offsets H2D failed: {e}")
                })?);
        }
        if self.active_counts.is_none() {
            self.active_counts =
                Some(ctx.stream.clone_htod(&[0i32]).map_err(|e| {
                    anyhow!("Qwen FP8 dense DeepGEMM active_counts H2D failed: {e}")
                })?);
        }
        Ok(())
    }
}

fn qwen_quant_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("ARLE_QWEN35_PROFILE").is_some()
            || std::env::var_os("ARLE_QWEN35_QUANT_PROFILE").is_some()
    })
}

fn qwen_fp8_deepgemm_dense_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if matches!(
            std::env::var("ARLE_QWEN35_DEEPGEMM").as_deref(),
            Ok("0" | "false" | "FALSE" | "no" | "off" | "OFF")
        ) {
            return false;
        }
        match cuda_moe::dsv4_deepgemm_native_preflight() {
            Ok(_) => true,
            Err(err) => {
                log::warn!("Qwen FP8 dense DeepGEMM disabled: native bridge unavailable ({err})");
                false
            }
        }
    })
}

/// DeepGEMM's dense FP8 GEMM is Hopper-only (`wgmma`, sm_90a). On pre-Hopper
/// GPUs (A100 = sm_80, V100 = sm_70) the native kernel returns
/// `CUDA_ERROR_NOT_SUPPORTED` at launch (`deepgemm_native.cu` hard-checks
/// `prop.major == 9`). Cache the device compute-capability major ONCE here so
/// the dense FP8 dispatch can SM-gate the DeepGEMM path without a per-step
/// `cuDeviceGetAttribute` (the known per-step device-property query perf bug).
/// On sm < 9 the caller routes the dense FP8 GEMM to the software-dequant →
/// BF16 cuBLAS path (large M) or the portable scalar block-scaled GEMV (small
/// M) — both run on sm_70+. The sm >= 9 path is byte-identical to before.
fn qwen_fp8_dense_sm_supports_deepgemm(ctx: &DeviceContext) -> bool {
    static SUPPORTS: OnceLock<bool> = OnceLock::new();
    *SUPPORTS.get_or_init(|| {
        let (major, minor) = ctx.compute_capability();
        let supports = major >= 9;
        if !supports {
            log::info!(
                "Qwen FP8 dense DeepGEMM SM-gated OFF on sm_{major}{minor} (Hopper sm_90 \
                 required for wgmma); using dequant→BF16 GEMM / scalar GEMV fallback"
            );
        }
        supports
    })
}

fn qwen_quant_profile<T>(
    ctx: &DeviceContext,
    label: &'static str,
    seq_len: usize,
    rows: usize,
    cols: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if !qwen_quant_profile_enabled() {
        return f();
    }
    let start = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let stop = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(&ctx.stream)?;
    let host_t0 = Instant::now();
    let result = f();
    let host_ms = host_t0.elapsed().as_secs_f64() * 1000.0;
    stop.record(&ctx.stream)?;
    stop.synchronize()?;
    let cuda_ms = start.elapsed_ms(&stop)? as f64;
    if std::env::var("INFER_TP_RANK")
        .map(|rank| rank == "0")
        .unwrap_or(true)
    {
        eprintln!(
            "[qwen-quant-profile] {label} seq={seq_len} rows={rows} cols={cols} cuda_ms={cuda_ms:.3} host_ms={host_ms:.3}"
        );
    }
    result
}

fn fp8_f32_scale_shape(weight: &DeviceMatrix) -> Result<(i32, i32, i32, i32)> {
    match weight.weight_format {
        WeightFormat::Fp8BlockScaled => {
            ensure!(
                weight.quant_scale_rows > 0
                    && weight.quant_scale_cols > 0
                    && weight.quant_block_m > 0
                    && weight.quant_block_k > 0,
                "fp8_block_scaled missing scale/block metadata: scale={}x{}, block={}x{}",
                weight.quant_scale_rows,
                weight.quant_scale_cols,
                weight.quant_block_m,
                weight.quant_block_k
            );
            Ok((
                weight.quant_scale_rows as i32,
                weight.quant_scale_cols as i32,
                weight.quant_block_m as i32,
                weight.quant_block_k as i32,
            ))
        }
        WeightFormat::Fp8PerShard => {
            ensure!(
                weight.quant_scale_rows == 1 && weight.quant_scale_cols == 1,
                "fp8_per_shard dispatch currently supports one resident shard scale, got {}x{}",
                weight.quant_scale_rows,
                weight.quant_scale_cols
            );
            Ok((1, 1, weight.rows as i32, weight.cols as i32))
        }
        other => Err(anyhow!(
            "expected FP8 f32-scale resident quant format, got {other}"
        )),
    }
}

fn fp8_deepgemm_dense_shape(ctx: &DeviceContext, weight: &DeviceMatrix, seq_len: usize) -> bool {
    weight.weight_format == WeightFormat::Fp8BlockScaled
        && seq_len >= QWEN_FP8_DEEPGEMM_DENSE_MIN_M
        && weight.quant_block_m == 128
        && weight.quant_block_k == 128
        && weight.rows.is_multiple_of(8)
        && weight.cols.is_multiple_of(128)
        && qwen_fp8_dense_sm_supports_deepgemm(ctx)
        && qwen_fp8_deepgemm_dense_enabled()
}

pub(super) fn warm_fp8_deepgemm_dense(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    seq_len: usize,
) -> Result<bool> {
    if !fp8_deepgemm_dense_shape(ctx, weight, seq_len) {
        return Ok(false);
    }
    let qw = weight
        .qweight_u8
        .as_ref()
        .ok_or_else(|| anyhow!("fp8_block_scaled missing qweight_u8"))?;
    let scales = weight
        .scale_f32
        .as_ref()
        .ok_or_else(|| anyhow!("fp8_block_scaled missing scale_f32"))?;
    let m = seq_len;
    let n = weight.rows;
    let k = weight.cols;
    let scale_stride_m = m.div_ceil(4) * 4;
    let scale_cols = k.div_ceil(128);
    let input_fp8 = ctx
        .stream
        .alloc_zeros::<u8>(m * k)
        .map_err(|e| anyhow!("Qwen FP8 dense DeepGEMM warm input alloc failed: {e}"))?;
    let input_scales = ctx
        .stream
        .alloc_zeros::<f32>(scale_stride_m * scale_cols)
        .map_err(|e| anyhow!("Qwen FP8 dense DeepGEMM warm scale alloc failed: {e}"))?;
    let out = ctx
        .stream
        .alloc_zeros::<bf16>(m * n)
        .map_err(|e| anyhow!("Qwen FP8 dense DeepGEMM warm output alloc failed: {e}"))?;
    qwen_quant_profile(ctx, "qwen/fp8/dense_deepgemm_warm", m, n, k, || unsafe {
        cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
            cache_ptr(&input_fp8, ctx),
            cache_ptr(&input_scales, ctx),
            cache_ptr(qw, ctx),
            cache_ptr(scales, ctx),
            cache_ptr(&out, ctx),
            m,
            n,
            k,
            scale_stride_m,
            ctx.stream.cu_stream(),
        )
    })?;
    Ok(true)
}

fn try_fp8_deepgemm_dense_batch(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<bool> {
    if !fp8_deepgemm_dense_shape(ctx, weight, x.seq_len) {
        return Ok(false);
    }
    let qw = weight
        .qweight_u8
        .as_ref()
        .ok_or_else(|| anyhow!("fp8_block_scaled missing qweight_u8"))?;
    let scales = weight
        .scale_f32
        .as_ref()
        .ok_or_else(|| anyhow!("fp8_block_scaled missing scale_f32"))?;
    let m = x.seq_len;
    let n = weight.rows;
    let k = weight.cols;
    let scale_stride_m = m.div_ceil(4) * 4;
    let scale_cols = k.div_ceil(128);
    QWEN_FP8_DENSE_SCRATCH.with(|cell| -> Result<()> {
        let mut scratch = cell.borrow_mut();
        scratch.ensure(ctx, m * k, scale_stride_m * scale_cols)?;
        {
            let active_counts = scratch
                .active_counts
                .as_mut()
                .ok_or_else(|| anyhow!("Qwen FP8 dense DeepGEMM active_counts missing"))?;
            ctx.stream
                .memcpy_htod(&[i32::try_from(m)?], active_counts)
                .map_err(|e| anyhow!("Qwen FP8 dense DeepGEMM active_counts H2D failed: {e}"))?;
        }
        let input_fp8 = scratch
            .input_fp8
            .as_ref()
            .ok_or_else(|| anyhow!("Qwen FP8 dense DeepGEMM input scratch missing"))?;
        let input_scales = scratch
            .input_scales
            .as_ref()
            .ok_or_else(|| anyhow!("Qwen FP8 dense DeepGEMM scale scratch missing"))?;
        let active_experts = scratch
            .active_experts
            .as_ref()
            .ok_or_else(|| anyhow!("Qwen FP8 dense DeepGEMM active_experts missing"))?;
        let active_offsets = scratch
            .active_offsets
            .as_ref()
            .ok_or_else(|| anyhow!("Qwen FP8 dense DeepGEMM active_offsets missing"))?;
        let active_counts = scratch
            .active_counts
            .as_ref()
            .ok_or_else(|| anyhow!("Qwen FP8 dense DeepGEMM active_counts missing"))?;

        qwen_quant_profile(ctx, "qwen/fp8/dense_pack_quantize", m, n, k, || unsafe {
            cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                cache_ptr(&x.data, ctx),
                cache_ptr(input_fp8, ctx),
                cache_ptr(input_scales, ctx),
                cache_ptr(active_experts, ctx),
                cache_ptr(active_offsets, ctx),
                cache_ptr(active_counts, ctx),
                1,
                m,
                k,
                scale_stride_m,
                ctx.stream.cu_stream(),
            )
        })?;
        qwen_quant_profile(ctx, "qwen/fp8/dense_deepgemm", m, n, k, || unsafe {
            cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
                cache_ptr(input_fp8, ctx),
                cache_ptr(input_scales, ctx),
                cache_ptr(qw, ctx),
                cache_ptr(scales, ctx),
                cache_ptr(&out.data, ctx),
                m,
                n,
                k,
                scale_stride_m,
                ctx.stream.cu_stream(),
            )
        })
    })?;
    Ok(true)
}

/// Pre-Hopper dense FP8 GEMM fallback for the DeepGEMM-shaped path: on sm < 9
/// (A100/V100), where DeepGEMM's `wgmma` kernel is unavailable, dequantize the
/// FP8 E4M3 block-scaled weight `[rows, cols]` to a resident BF16 scratch ONCE,
/// then run the existing BF16 cuBLAS GEMM (`gemm_cuda`). This is the large-M
/// (prefill) path: dequant is `rows*cols` work amortized over the `M` GEMM rows,
/// far cheaper than the warp-per-row scalar GEMV at large M.
///
/// Engages for large-M (prefill) on ANY arch whenever DeepGEMM did not handle the
/// GEMM — we only reach this fn *after* `try_fp8_deepgemm_dense_batch` returned
/// false (DeepGEMM disabled/unbuilt, or it declined the shape). Prefill must NEVER
/// fall through to the scalar GEMV below — that is a memory-bound per-token path
/// (~20× slower at M=2048): dequant once, cuBLAS GEMM over all M rows instead.
///
/// Only engages when:
///  - the weight is `Fp8BlockScaled` with the canonical 128x128 block shape,
///  - `M >= QWEN_FP8_DEEPGEMM_DENSE_MIN_M` (small-M decode keeps the scalar GEMV,
///    which is already coalesced + occupancy-friendly there).
///
/// Returns `Ok(false)` when it does not apply (small-M decode), leaving
/// `gemm_batch` to fall through to the scalar/MMA block-scaled GEMV.
fn try_fp8_dequant_bf16_gemm_batch(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<bool> {
    if weight.weight_format != WeightFormat::Fp8BlockScaled
        || x.seq_len < QWEN_FP8_DEEPGEMM_DENSE_MIN_M
        || weight.quant_block_m != 128
        || weight.quant_block_k != 128
    {
        return Ok(false);
    }
    let qw = weight
        .qweight_u8
        .as_ref()
        .ok_or_else(|| anyhow!("fp8_block_scaled missing qweight_u8"))?;
    let scales = weight
        .scale_f32
        .as_ref()
        .ok_or_else(|| anyhow!("fp8_block_scaled missing scale_f32"))?;
    let (scale_rows, scale_cols, block_m, block_k) = fp8_f32_scale_shape(weight)?;
    let n = weight.rows; // GEMM M dim (weight rows)
    let k = weight.cols; // GEMM K dim (contraction)
    let weight_elems = n * k;

    QWEN_FP8_DEQUANT_SCRATCH.with(|cell| -> Result<()> {
        let mut scratch = cell.borrow_mut();
        if scratch.capacity < weight_elems {
            scratch.weight_bf16 =
                Some(ctx.stream.alloc_zeros::<bf16>(weight_elems).map_err(|e| {
                    anyhow!("Qwen FP8 dense dequant BF16 scratch alloc failed: {e}")
                })?);
            scratch.capacity = weight_elems;
        }
        let weight_bf16 = scratch
            .weight_bf16
            .as_ref()
            .ok_or_else(|| anyhow!("Qwen FP8 dense dequant BF16 scratch missing"))?;
        let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
        let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
        let (wbf16_ptr, _gw) = weight_bf16.device_ptr(&ctx.stream);
        let stream = ctx.stream.cu_stream();
        qwen_quant_profile(
            ctx,
            "qwen/fp8/dense_dequant_bf16",
            x.seq_len,
            n,
            k,
            || unsafe {
                ffi::dequantize_fp8_block_scaled_to_bf16_cuda(
                    qw_ptr as *const u8,
                    scales_ptr as *const f32,
                    wbf16_ptr as *mut ffi::Half,
                    n as i32,
                    k as i32,
                    scale_rows,
                    scale_cols,
                    block_m,
                    block_k,
                    stream,
                )
                .result()
                .map_err(|e| anyhow!("Qwen FP8 dense dequant kernel failed: {e}"))
            },
        )?;
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
        qwen_quant_profile(
            ctx,
            "qwen/fp8/dense_dequant_gemm",
            x.seq_len,
            n,
            k,
            || unsafe {
                ffi::gemm_cuda(
                    wbf16_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    n as i32,
                    x.seq_len as i32,
                    k as i32,
                    stream,
                )
                .result()
                .map_err(|e| anyhow!("Qwen FP8 dense dequant BF16 GEMM failed: {e}"))
            },
        )
    })?;
    Ok(true)
}

pub(super) fn gemm_batch(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    if matches!(
        weight.weight_format,
        WeightFormat::Fp8BlockScaled | WeightFormat::Fp8PerShard
    ) && try_fp8_deepgemm_dense_batch(ctx, weight, x, out)?
    {
        return Ok(());
    }

    // Pre-Hopper (sm < 9) large-M dense FP8: DeepGEMM's wgmma path is gated off
    // above, so dequant → BF16 cuBLAS GEMM here (small-M decode keeps the scalar
    // GEMV below). No-op on Hopper (returns false).
    if try_fp8_dequant_bf16_gemm_batch(ctx, weight, x, out)? {
        return Ok(());
    }

    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();

    unsafe {
        match weight.weight_format {
            WeightFormat::Dsv4Fp8BlockScaled | WeightFormat::Dsv4Fp4BlockScaled => {
                let qw = weight
                    .qweight
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing qweight", weight.weight_format))?;
                let scales = weight
                    .dsv4_scales
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing dsv4_scales", weight.weight_format))?;
                let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
                let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
                let res = match weight.weight_format {
                    WeightFormat::Dsv4Fp8BlockScaled => ffi::dsv4_fp8_gemv_batch_cuda(
                        qw_ptr as *const u8,
                        scales_ptr as *const u8,
                        x_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        x.seq_len as i32,
                        weight.rows as i32,
                        weight.cols as i32,
                        weight.dsv4_scale_rows as i32,
                        weight.dsv4_scale_cols as i32,
                        stream,
                    ),
                    WeightFormat::Dsv4Fp4BlockScaled => ffi::dsv4_fp4_gemv_batch_cuda(
                        qw_ptr as *const u8,
                        scales_ptr as *const u8,
                        x_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        x.seq_len as i32,
                        weight.rows as i32,
                        weight.cols as i32,
                        weight.dsv4_scale_rows as i32,
                        weight.dsv4_scale_cols as i32,
                        stream,
                    ),
                    _ => unreachable!(),
                };
                res.result()?;
            }
            WeightFormat::Fp8BlockScaled | WeightFormat::Fp8PerShard => {
                let qw = weight
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing qweight_u8", weight.weight_format))?;
                let scales = weight
                    .scale_f32
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing scale_f32", weight.weight_format))?;
                let (scale_rows, scale_cols, block_m, block_k) = fp8_f32_scale_shape(weight)?;
                let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
                let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
                // The coalesced scalar warp-per-row GEMV is the production path:
                // 3.6x faster than the tensor-core MMA tile at B=1 decode on H20
                // (the MMA was occupancy-starved + uncoalesced off its batched
                // B<=16 design point — KILLED, see
                // docs/experience/wins/2026-06-22-qwen-fp8-decode-gemv-scalar.md).
                qwen_quant_profile(
                    ctx,
                    "qwen/fp8/gemv_batch",
                    x.seq_len,
                    weight.rows,
                    weight.cols,
                    || {
                        Ok(ffi::gemv_fp8_block_scaled_batch_cuda(
                            qw_ptr as *const u8,
                            scales_ptr as *const f32,
                            x_ptr as *const ffi::Half,
                            out_ptr as *mut ffi::Half,
                            x.seq_len as i32,
                            weight.rows as i32,
                            weight.cols as i32,
                            scale_rows,
                            scale_cols,
                            block_m,
                            block_k,
                            stream,
                        )
                        .result()?)
                    },
                )?;
            }
            WeightFormat::Fp4E2M1Group => {
                ensure!(
                    weight.quant_scale_rows == weight.rows,
                    "fp4_e2m1_group scale rows {} != weight rows {}",
                    weight.quant_scale_rows,
                    weight.rows
                );
                ensure!(
                    weight.group_size > 0
                        && weight.quant_scale_cols == weight.cols / weight.group_size,
                    "fp4_e2m1_group scale cols {} incompatible with cols {} group_size {}",
                    weight.quant_scale_cols,
                    weight.cols,
                    weight.group_size
                );
                let qw = weight
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("fp4_e2m1_group missing qweight_u8"))?;
                let scales = weight
                    .qscale_fp8
                    .as_ref()
                    .ok_or_else(|| anyhow!("fp4_e2m1_group missing qscale_fp8"))?;
                let global = weight
                    .scale_f32
                    .as_ref()
                    .ok_or_else(|| anyhow!("fp4_e2m1_group missing scale_f32 global scale"))?;
                ensure!(
                    global.len() == 1,
                    "fp4_e2m1_group dispatch currently supports one global scale, got {}",
                    global.len()
                );
                let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
                let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
                let (global_ptr, _gg) = global.device_ptr(&ctx.stream);
                ffi::gemv_fp4_e2m1_group_batch_cuda(
                    qw_ptr as *const u8,
                    scales_ptr as *const u8,
                    global_ptr as *const f32,
                    x_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    x.seq_len as i32,
                    weight.rows as i32,
                    weight.cols as i32,
                    weight.group_size as i32,
                    weight.quant_scale_cols as i32,
                    stream,
                )
                .result()?;
            }
            other => bail!("gemm_batch unsupported resident quant weight format {other}"),
        }
    }
    Ok(())
}

pub(super) fn gemv(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &DeviceVec,
    out: &mut DeviceVec,
) -> Result<()> {
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();

    unsafe {
        match weight.weight_format {
            WeightFormat::Fp8BlockScaled | WeightFormat::Fp8PerShard => {
                let qw = weight
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing qweight_u8", weight.weight_format))?;
                let scales = weight
                    .scale_f32
                    .as_ref()
                    .ok_or_else(|| anyhow!("{} missing scale_f32", weight.weight_format))?;
                let (scale_rows, scale_cols, block_m, block_k) = fp8_f32_scale_shape(weight)?;
                let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
                let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
                ffi::gemv_fp8_block_scaled_cuda(
                    qw_ptr as *const u8,
                    scales_ptr as *const f32,
                    x_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    weight.rows as i32,
                    weight.cols as i32,
                    scale_rows,
                    scale_cols,
                    block_m,
                    block_k,
                    stream,
                )
                .result()?;
            }
            WeightFormat::Fp4E2M1Group => {
                ensure!(
                    weight.quant_scale_rows == weight.rows,
                    "fp4_e2m1_group scale rows {} != weight rows {}",
                    weight.quant_scale_rows,
                    weight.rows
                );
                ensure!(
                    weight.group_size > 0
                        && weight.quant_scale_cols == weight.cols / weight.group_size,
                    "fp4_e2m1_group scale cols {} incompatible with cols {} group_size {}",
                    weight.quant_scale_cols,
                    weight.cols,
                    weight.group_size
                );
                let qw = weight
                    .qweight_u8
                    .as_ref()
                    .ok_or_else(|| anyhow!("fp4_e2m1_group missing qweight_u8"))?;
                let scales = weight
                    .qscale_fp8
                    .as_ref()
                    .ok_or_else(|| anyhow!("fp4_e2m1_group missing qscale_fp8"))?;
                let global = weight
                    .scale_f32
                    .as_ref()
                    .ok_or_else(|| anyhow!("fp4_e2m1_group missing scale_f32 global scale"))?;
                ensure!(
                    global.len() == 1,
                    "fp4_e2m1_group dispatch currently supports one global scale, got {}",
                    global.len()
                );
                let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
                let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
                let (global_ptr, _gg) = global.device_ptr(&ctx.stream);
                ffi::gemv_fp4_e2m1_group_cuda(
                    qw_ptr as *const u8,
                    scales_ptr as *const u8,
                    global_ptr as *const f32,
                    x_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    weight.rows as i32,
                    weight.cols as i32,
                    weight.group_size as i32,
                    weight.quant_scale_cols as i32,
                    stream,
                )
                .result()?;
            }
            other => bail!("gemv unsupported resident quant weight format {other}"),
        }
    }
    Ok(())
}
