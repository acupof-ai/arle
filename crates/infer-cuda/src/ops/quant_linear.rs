use anyhow::{Result, anyhow, bail, ensure};
use cuda_kernels::ffi;
use cuda_kernels::moe as cuda_moe;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::tensor::{WeightFormat, cache_ptr};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, sys::CUevent_flags};
use half::bf16;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

mod qwen_fp8_dense_policy {
    include!("generated/qwen_fp8_dense_projection.rs");
}

/// Pre-Hopper dequant→BF16-cuBLAS floor. Stays at the old 16: that path pays a
/// full weight dequant (1 read + 2 bf16 writes) per call, so tiny M belongs on
/// the GEMV there even though Hopper's DeepGEMM lane wins from M=2.
const QWEN_FP8_DEQUANT_GEMM_MIN_M: usize = 16;

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
    // Cache of dequantized FP16 weights, keyed by the original quantized weight
    // pointer. Only used for W4A16 prefill on sm<9: dequant is the dominant
    // cost, so caching the smaller projections (attention qkv/o) avoids
    // re-dequantizing them on every prefill. Larger MLP weights are not cached
    // (they would exceed VRAM).
    w4a16_fp16_cache: HashMap<u64, CudaSlice<bf16>>,
}

thread_local! {
    static QWEN_FP8_DEQUANT_SCRATCH: RefCell<QwenFp8DequantScratch> =
        RefCell::new(QwenFp8DequantScratch::default());
}

thread_local! {
    static QWEN_FP8_DENSE_SCRATCH: RefCell<QwenFp8DenseScratch> =
        RefCell::new(QwenFp8DenseScratch::default());
}

/// Reusable Marlin W8A16 GEMM scratch: fp32-reduce `c_tmp` + int lock
/// `workspace`. Both sizes depend only on the device SM count (constant), so
/// they are allocated ONCE and never grow — the Qwen decode loop is CUDA-graph
/// captured, and a per-call `cudaMalloc` would break capture (the FP8 paths use
/// the same pre-alloc discipline). `workspace` is zeroed once at allocation
/// (Marlin leaves the locks at 0 after each GEMM, so reuse is safe).
#[derive(Default)]
struct MarlinW8a16Scratch {
    c_tmp: Option<CudaSlice<f32>>,
    workspace: Option<CudaSlice<i32>>,
}

thread_local! {
    static MARLIN_W8A16_SCRATCH: RefCell<MarlinW8a16Scratch> =
        RefCell::new(MarlinW8a16Scratch::default());
}

// Qwen FP8 dense dispatch counters. Per-family split if/when more operator
// families migrate to generated policy. fallback_count is derived (gemv +
// dequant), not an independent atomic.
static DEEPGEMM_HITS: AtomicU64 = AtomicU64::new(0);
static GEMV_HITS: AtomicU64 = AtomicU64::new(0);
static DEQUANT_GEMM_HITS: AtomicU64 = AtomicU64::new(0);
static MARLIN_W8A16_HITS: AtomicU64 = AtomicU64::new(0);

static FP8_IMPLEMENTATION_IDS: &[(&AtomicU64, &str)] = &[
    (&DEEPGEMM_HITS, "cuda.qwen.fp8_pack_deepgemm"),
    (&GEMV_HITS, "cuda.qwen.fp8_gemv"),
    (&DEQUANT_GEMM_HITS, "cuda.qwen.fp8_dequant_bf16_gemm"),
    (&MARLIN_W8A16_HITS, "cuda.w8a16.marlin_tensorcore"),
];

/// Cumulative operator dispatch stats for Qwen FP8 dense projection.
///
/// Materialized only at an explicit stats request boundary. Dispatch itself only
/// increments atomics; no request or engine-tick path allocates telemetry data.
pub(crate) fn qwen_fp8_dense_operator_stats() -> infer_seam::OperatorDispatchStats {
    use infer_seam::OperatorImplementationHits;

    let implementation_hits: Vec<_> = FP8_IMPLEMENTATION_IDS
        .iter()
        .filter_map(|(counter, id)| {
            let hits = counter.load(Ordering::Relaxed);
            (hits > 0).then(|| OperatorImplementationHits {
                implementation_id: (*id).into(),
                hits,
            })
        })
        .collect();
    let fallback_count =
        GEMV_HITS.load(Ordering::Relaxed) + DEQUANT_GEMM_HITS.load(Ordering::Relaxed);

    infer_seam::OperatorDispatchStats {
        policy_hash: qwen_fp8_dense_policy::POLICY_ID.into(),
        implementation_hits,
        fallback_count,
    }
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
        if !crate::runtime_flags::qwen35_deepgemm() {
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
/// On any non-Hopper SM the caller routes the dense FP8 GEMM to the
/// software-dequant → BF16 cuBLAS path (large M) or the portable scalar
/// block-scaled GEMV (small M) — both run on sm_70+. The Hopper path is
/// byte-identical to before.
fn qwen_fp8_dense_sm_supports_deepgemm(ctx: &DeviceContext) -> bool {
    static SUPPORTS: OnceLock<bool> = OnceLock::new();
    *SUPPORTS.get_or_init(|| {
        let (major, minor) = ctx.compute_capability();
        // DeepGEMM is Hopper-exclusive: deepgemm_native.cu refuses major != 9
        // at every entry. Blackwell (major 10/12) must fall to the portable path.
        let supports = major == 9;
        if !supports {
            log::info!(
                "Qwen FP8 dense DeepGEMM SM-gated OFF on sm_{major}{minor} (Hopper sm_90 \
                 required for wgmma); using dequant→BF16 GEMM / scalar GEMV fallback"
            );
        }
        supports
    })
}

/// W8A16 Marlin tensor-core GEMM is Ampere+ (`mma.sync.m16n8k16` + `cp.async`,
/// gated `#if __CUDA_ARCH__ < 800` to no-op stubs in the vendored kernels). One
/// binary runs sm_80..sm_120. Below sm_80 the shim returns NOT_SUPPORTED; cache
/// the gate ONCE so decode dispatch avoids a per-step `cuDeviceGetAttribute`.
/// When off, W8A16 keeps the dequant→BF16 GEMM (large M) / scalar GEMV (small M).
fn w8a16_sm_supports_marlin(ctx: &DeviceContext) -> bool {
    static SUPPORTS: OnceLock<bool> = OnceLock::new();
    *SUPPORTS.get_or_init(|| {
        let (major, _minor) = ctx.compute_capability();
        let supports = major >= 8;
        if !supports {
            log::info!(
                "W8A16 Marlin SM-gated OFF on sm_{major}x (Ampere sm_80+ required for \
                 mma.sync tensor cores); using dequant→BF16 GEMM / scalar GEMV fallback"
            );
        }
        supports
    })
}

fn qwen_fp8_dense_route(
    ctx: &DeviceContext,
    m: usize,
    n: usize,
    k: usize,
) -> qwen_fp8_dense_policy::Route {
    if !qwen_fp8_dense_policy::HAS_EXACT_CELLS {
        return qwen_fp8_dense_policy::fallback(m);
    }
    static HARDWARE: OnceLock<(i32, i32, usize)> = OnceLock::new();
    let &(sm_major, sm_minor, sm_count) = HARDWARE.get_or_init(|| {
        let (major, minor) = ctx.compute_capability();
        (major, minor, ctx.sm_count())
    });
    qwen_fp8_dense_policy::select_exact(sm_major, sm_minor, sm_count, m, n, k)
        .unwrap_or_else(|| qwen_fp8_dense_policy::fallback(m))
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
        && qwen_fp8_dense_route(ctx, seq_len, weight.rows, weight.cols)
            == qwen_fp8_dense_policy::Route::PackDeepGemm
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
    // SAFETY: ptrs from live device allocations sized to the dims passed.
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

        // SAFETY: ptrs from live device allocations sized to the dims passed.
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
        // SAFETY: ptrs from live device allocations sized to the dims passed.
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
///  - `M >= QWEN_FP8_DEQUANT_GEMM_MIN_M` (small-M decode keeps the scalar GEMV:
///    a full-weight dequant per call is never worth it at tiny M).
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
        || x.seq_len < QWEN_FP8_DEQUANT_GEMM_MIN_M
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
            // SAFETY: ptrs from live device allocations sized to the dims passed.
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
            // SAFETY: ptrs from live device allocations sized to the dims passed.
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

/// W8A16 Marlin tensor-core GEMM (Ampere+): C[m,n] = X[m,k] @ dequant(W). Fires
/// when the SM gate is on AND the weight was Marlin-repacked at load
/// (`marlin_packed`/`marlin_scales` present, set by `repack_for_marlin_w8a16`).
/// Supersedes both the scalar batched-GEMV (decode) and the dequant→cuBLAS
/// fallback (prefill) — the win is tensor cores in decode, ~bf16 speed at INT8
/// VRAM. Returns false (→ existing fallbacks) when off or not prepacked.
fn try_w8a16_marlin_gemm_batch(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<bool> {
    if weight.weight_format != WeightFormat::W8A16 || !w8a16_sm_supports_marlin(ctx) {
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
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();
    MARLIN_W8A16_SCRATCH.with(|cell| -> Result<()> {
        let mut scratch = cell.borrow_mut();
        // Allocate once at the SM-derived MAX (c_tmp caps at m=64; workspace is
        // m-independent). Never grows → graph-capture safe. Zero the workspace at
        // alloc; Marlin resets its locks to 0 after each GEMM, so reuse is safe.
        if scratch.c_tmp.is_none() {
            let sms = ctx.sm_count() as i32;
            // SAFETY: pure size queries (arithmetic on sms), no device work.
            let c_tmp_floats = unsafe { ffi::marlin_w8a16_c_tmp_floats(64, sms) } as usize;
            // SAFETY: pure size query, no device work.
            let ws_ints = unsafe { ffi::marlin_w8a16_workspace_ints(sms) } as usize;
            scratch.c_tmp = Some(
                ctx.stream
                    .alloc_zeros::<f32>(c_tmp_floats)
                    .map_err(|e| anyhow!("Marlin W8A16 c_tmp alloc failed: {e}"))?,
            );
            scratch.workspace = Some(
                ctx.stream
                    .alloc_zeros::<i32>(ws_ints)
                    .map_err(|e| anyhow!("Marlin W8A16 workspace alloc failed: {e}"))?,
            );
        }
        let c_tmp = scratch.c_tmp.as_ref().unwrap();
        let workspace = scratch.workspace.as_ref().unwrap();
        let (c_tmp_ptr, _gc) = c_tmp.device_ptr(&ctx.stream);
        let (ws_ptr, _gw) = workspace.device_ptr(&ctx.stream);
        qwen_quant_profile(ctx, "qwen/w8a16/marlin_gemm", x.seq_len, n, k, || {
            // SAFETY: all ptrs from live device allocations; packed/scales sized by
            // repack_for_marlin_w8a16 for these dims, x=[seq_len,k], out=[seq_len,n],
            // c_tmp/workspace sized to the SM max above.
            unsafe {
                ffi::marlin_w8a16_gemm_cuda(
                    x_ptr as *const ffi::Half,
                    packed_ptr as *const u32,
                    scales_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    c_tmp_ptr as *mut f32,
                    ws_ptr as *mut i32,
                    x.seq_len as i32,
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
/// GEMM over all M rows. Mirrors `try_fp8_dequant_bf16_gemm_batch` — the scalar
/// GEMV re-reads the weight per token (~20× slower at M=2048 and the cause of
/// W8A16's 6× TTFT), so prefill must dequant once and GEMM instead. Small-M
/// decode (`M < QWEN_FP8_DEQUANT_GEMM_MIN_M`) keeps the batched GEMV.
fn try_w8a16_dequant_bf16_gemm_batch(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<bool> {
    if weight.weight_format != WeightFormat::W8A16 || x.seq_len < QWEN_FP8_DEQUANT_GEMM_MIN_M {
        return Ok(false);
    }
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
    QWEN_FP8_DEQUANT_SCRATCH.with(|cell| -> Result<()> {
        let mut scratch = cell.borrow_mut();
        if scratch.capacity < weight_elems {
            scratch.weight_bf16 =
                Some(ctx.stream.alloc_zeros::<bf16>(weight_elems).map_err(|e| {
                    anyhow!("Qwen W8A16 dense dequant BF16 scratch alloc failed: {e}")
                })?);
            scratch.capacity = weight_elems;
        }
        let weight_bf16 = scratch
            .weight_bf16
            .as_ref()
            .ok_or_else(|| anyhow!("Qwen W8A16 dense dequant BF16 scratch missing"))?;
        let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
        let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
        let (wbf16_ptr, _gw) = weight_bf16.device_ptr(&ctx.stream);
        let stream = ctx.stream.cu_stream();
        qwen_quant_profile(
            ctx,
            "qwen/w8a16/dense_dequant_bf16",
            x.seq_len,
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
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
        qwen_quant_profile(
            ctx,
            "qwen/w8a16/dense_dequant_gemm",
            x.seq_len,
            n,
            k,
            // SAFETY: ptrs from live device allocations sized to the dims passed.
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
                .map_err(|e| anyhow!("Qwen W8A16 dense dequant BF16 GEMM failed: {e}"))
            },
        )?;
        Ok(())
    })?;
    Ok(true)
}

fn try_w4a16_dequant_bf16_gemm_batch(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<bool> {
    if weight.weight_format != WeightFormat::W4A16 || x.seq_len < QWEN_FP8_DEQUANT_GEMM_MIN_M {
        return Ok(false);
    }
    ensure!(
        weight.group_size > 0 && weight.cols.is_multiple_of(weight.group_size),
        "W4A16 cols {} not group-aligned to {}",
        weight.cols,
        weight.group_size
    );
    let qw = weight
        .qweight
        .as_ref()
        .ok_or_else(|| anyhow!("W4A16 missing qweight"))?;
    let scales = weight
        .qscales
        .as_ref()
        .ok_or_else(|| anyhow!("W4A16 missing qscales"))?;
    let n = weight.rows;
    let k = weight.cols;
    let weight_elems = n * k;
    // Cache dequantized FP16 weights for small projections only. Larger QKV/MLP
    // weights are dequantized per-call (caching them would OOM on V100's 32GB).
    const W4A16_CACHE_MAX_ELEMS: usize = 10_000_000;

    let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
    let qw_ptr_u64 = qw_ptr as u64;
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();

    QWEN_FP8_DEQUANT_SCRATCH.with(|cell| -> Result<()> {
        let mut scratch = cell.borrow_mut();

        // Try the cache first (small weights only).
        if weight_elems <= W4A16_CACHE_MAX_ELEMS {
            if let Some(cached) = scratch.w4a16_fp16_cache.get(&qw_ptr_u64) {
                let (wfp16_ptr, _gw) = cached.device_ptr(&ctx.stream);
                // SAFETY: all pointers are valid device buffers from the context, sizes match the GEMM dims.
                unsafe {
                    ffi::gemm_fp16_weight_cuda(
                        wfp16_ptr as *const ffi::Half,
                        x_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        n as i32,
                        x.seq_len as i32,
                        k as i32,
                        stream,
                    )
                    .result()
                    .map_err(|e| anyhow!("W4A16 cached FP16 GEMM failed: {e}"))?;
                }
                return Ok(());
            }
        }

        // Dequant into the scratch (or a cached allocation).
        let weight_fp16: &CudaSlice<bf16> = if weight_elems <= W4A16_CACHE_MAX_ELEMS {
            let buf = ctx
                .stream
                .alloc_zeros::<bf16>(weight_elems)
                .map_err(|e| anyhow!("W4A16 dense dequant FP16 cache alloc failed: {e}"))?;
            {
                let (wfp16_ptr, _gw) = buf.device_ptr(&ctx.stream);
                // SAFETY: all pointers are valid device buffers, sizes match the dequant kernel dims.
                unsafe {
                    ffi::dequantize_w4a16_to_fp16_cuda(
                        qw_ptr as *const u8,
                        scales_ptr as *const ffi::Half,
                        wfp16_ptr as *mut ffi::Half,
                        n as i32,
                        k as i32,
                        weight.group_size as i32,
                        stream,
                    )
                    .result()
                    .map_err(|e| anyhow!("W4A16 dense dequant FP16 kernel failed: {e}"))?;
                }
            }
            scratch.w4a16_fp16_cache.insert(qw_ptr_u64, buf);
            scratch.w4a16_fp16_cache.get(&qw_ptr_u64).unwrap()
        } else {
            if scratch.capacity < weight_elems {
                scratch.weight_bf16 =
                    Some(ctx.stream.alloc_zeros::<bf16>(weight_elems).map_err(|e| {
                        anyhow!("W4A16 dense dequant FP16 scratch alloc failed: {e}")
                    })?);
                scratch.capacity = weight_elems;
            }
            let buf = scratch
                .weight_bf16
                .as_ref()
                .ok_or_else(|| anyhow!("W4A16 dense dequant FP16 scratch missing"))?;
            {
                let (wfp16_ptr, _gw) = buf.device_ptr(&ctx.stream);
                // SAFETY: all pointers are valid device buffers, sizes match the dequant kernel dims.
                unsafe {
                    ffi::dequantize_w4a16_to_fp16_cuda(
                        qw_ptr as *const u8,
                        scales_ptr as *const ffi::Half,
                        wfp16_ptr as *mut ffi::Half,
                        n as i32,
                        k as i32,
                        weight.group_size as i32,
                        stream,
                    )
                    .result()
                    .map_err(|e| anyhow!("W4A16 dense dequant FP16 kernel failed: {e}"))?;
                }
            }
            buf
        };

        let (wfp16_ptr, _gw) = weight_fp16.device_ptr(&ctx.stream);
        // SAFETY: all pointers are valid device buffers, sizes match the GEMM dims.
        unsafe {
            ffi::gemm_fp16_weight_cuda(
                wfp16_ptr as *const ffi::Half,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                n as i32,
                x.seq_len as i32,
                k as i32,
                stream,
            )
            .result()
            .map_err(|e| anyhow!("W4A16 dense dequant FP16 GEMM failed: {e}"))?;
        }
        Ok(())
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
        DEEPGEMM_HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    }

    // Pre-Hopper (sm < 9) large-M dense FP8: DeepGEMM's wgmma path is gated off
    // above, so dequant → BF16 cuBLAS GEMM here (small-M decode keeps the scalar
    // GEMV below). No-op on Hopper (returns false).
    if try_fp8_dequant_bf16_gemm_batch(ctx, weight, x, out)? {
        DEQUANT_GEMM_HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    }

    // W8A16 tensor-core (Ampere+): Marlin GEMM when the weight was repacked at
    // load. Supersedes both the dequant→cuBLAS prefill fallback and the scalar
    // batched-GEMV below. Returns false on pre-sm_80 / unaligned (not repacked).
    if try_w8a16_marlin_gemm_batch(ctx, weight, x, out)? {
        return Ok(());
    }

    // W8A16 large-M (prefill): dequant INT8→BF16 once + one cuBLAS GEMM, instead
    // of the per-token weight re-read of the batched GEMV below. Small-M decode
    // returns false and keeps the GEMV/batched-GEMM path.
    if try_w8a16_dequant_bf16_gemm_batch(ctx, weight, x, out)? {
        DEQUANT_GEMM_HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    }

    // W4A16 large-M (prefill): dequant INT4→BF16 once + one cuBLAS GEMM.
    // Same rationale as W8A16: the on-the-fly dequant GEMM uses FP32 math
    // (no tensor cores on V100), so dequant + BF16 GEMM (FP16 tensor cores
    // on sm_80+, FP16-cast on sm_70) is much faster for prefill.
    if try_w4a16_dequant_bf16_gemm_batch(ctx, weight, x, out)? {
        DEQUANT_GEMM_HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    }

    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();

    // SAFETY: ptrs from live device allocations sized to the dims passed.
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
                GEMV_HITS.fetch_add(1, Ordering::Relaxed);
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
            WeightFormat::W4A16 => {
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
                ffi::w4a16_gemv_batch_cuda(
                    qw_ptr as *const u8,
                    scales_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    x.seq_len as i32,
                    weight.rows as i32,
                    weight.cols as i32,
                    weight.group_size as i32,
                    stream,
                )
                .result()?;
                GEMV_HITS.fetch_add(1, Ordering::Relaxed);
            }
            WeightFormat::W8A16 => {
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
                ffi::w8a16_gemv_batch_cuda(
                    qw_ptr as *const i8,
                    scales_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    x.seq_len as i32,
                    weight.rows as i32,
                    weight.cols as i32,
                    weight.group_size as i32,
                    stream,
                )
                .result()?;
                GEMV_HITS.fetch_add(1, Ordering::Relaxed);
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

    // SAFETY: ptrs from live device allocations sized to the dims passed.
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
                GEMV_HITS.fetch_add(1, Ordering::Relaxed);
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
            // DSv4 block-scaled resident weights at M=1 (the DSpark draft head's
            // projections run seq_len=1). The base decode routes these through the
            // pre-repacked DeepGEMM cache; the draft weights aren't repacked, so
            // they land here — dispatch to the same batched resident kernel
            // `gemm_batch` uses, with batch=1.
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
                        1,
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
                        1,
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
            WeightFormat::W4A16 => {
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
                GEMV_HITS.fetch_add(1, Ordering::Relaxed);
            }
            WeightFormat::W8A16 => {
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
                GEMV_HITS.fetch_add(1, Ordering::Relaxed);
            }
            other => bail!("gemv unsupported resident quant weight format {other}"),
        }
    }
    Ok(())
}
