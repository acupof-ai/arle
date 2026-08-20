use anyhow::{Result, anyhow, ensure};
use cuda_kernels::ffi;
use cuda_kernels::moe as cuda_moe;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix};
use cuda_kernels::tensor::{RawDevicePtr, WeightFormat, cache_ptr};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    QWEN_DEQUANT_GEMM_PREFILL_MIN_M, QWEN_FP8_DEEPGEMM_PER_CHANNEL_MIN_M, marlin_sm_supported,
    qwen_quant_profile, with_marlin_scratch,
};

// The dense-route policy; POLICY_ID is consumed by the stats aggregation in
// the parent module's own inclusion, hence the dead-code allow here.
mod qwen_fp8_dense_policy {
    #![allow(dead_code)]
    include!("generated/qwen_fp8_dense_projection.rs");
}

pub(super) static DEEPGEMM_HITS: AtomicU64 = AtomicU64::new(0);
pub(super) static DEEPGEMM_PER_CHANNEL_HITS: AtomicU64 = AtomicU64::new(0);
pub(super) static FP8_DEQUANT_GEMM_HITS: AtomicU64 = AtomicU64::new(0);
pub(super) static MARLIN_FP8_HITS: AtomicU64 = AtomicU64::new(0);
pub(super) static FP8_GEMV_HITS: AtomicU64 = AtomicU64::new(0);

/// The M at or above which a per-channel FP8 weight goes to DeepGEMM, or `None`
/// when this engine can never reach it.
///
/// The arm is a prefill trade — DeepGEMM's 0.182 ms floor and this arm's
/// per-call H2D lose to Marlin at every decode row count measured — but
/// `gemm_batch` is handed a row count, not a phase, and a batched decode step's
/// row count IS the concurrency. So the floor sits above the largest decode
/// batch the engine can build (`--max-running-requests`, else the `num_slots`
/// ceiling), not above the 256 that happens to be the default. `None` when the
/// envelope is undeclared, and when the engine's own prefill chunk is shorter
/// than the floor (`--low-impact` pins it to 32, and `--chunked-prefill-size`
/// takes 128) — there the arm can never fire and Marlin owns every M.
pub(super) fn dense_deepgemm_prefill_floor(base: usize) -> Option<usize> {
    let (decode_rows, prefill_rows) = crate::runtime_flags::dense_gemm_row_envelope()?;
    let floor = base.max(decode_rows + 1);
    (prefill_rows >= floor).then_some(floor)
}

#[derive(Default)]
pub(super) struct QwenFp8DenseScratch {
    input_fp8: Option<CudaSlice<u8>>,
    input_fp8_capacity: usize,
    input_scales: Option<CudaSlice<f32>>,
    input_scales_capacity: usize,
    active_experts: Option<CudaSlice<i32>>,
    active_offsets: Option<CudaSlice<i32>>,
    active_counts: Option<CudaSlice<i32>>,
    sfb_ones: Option<CudaSlice<f32>>,
    sfb_ones_capacity: usize,
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
    /// The DeepGEMM prefill arms' B operand, built from the Marlin layout per
    /// call: NVFP4 widened to E4M3, or per-channel FP8 un-permuted. One buffer
    /// serves both — they run one weight at a time, and `[n, k]` E4M3 is the
    /// same shape either way.
    weight_fp8: Option<CudaSlice<u8>>,
    fp8_capacity: usize,
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

    /// The all-ones `sfb` the per-channel arm contracts against, grown to the
    /// largest per-channel shape this thread has seen. Every entry is 1.0, so
    /// one buffer serves every weight as long as it is long enough.
    fn ensure_sfb_ones(&mut self, ctx: &DeviceContext, n: usize, k: usize) -> Result<()> {
        let len = sfb_ones_len(n, k);
        if self.sfb_ones_capacity >= len {
            return Ok(());
        }
        self.sfb_ones = Some(
            ctx.stream
                .clone_htod(&vec![1.0f32; len])
                .map_err(|e| anyhow!("Qwen FP8 dense DeepGEMM sfb ones H2D failed: {e}"))?,
        );
        self.sfb_ones_capacity = len;
        Ok(())
    }
}

/// DeepGEMM indexes `sfb` as a `[ceil(n/128), ceil(k/128)]` K-major f32 grid
/// (`sm90_fp8_gemm_1d2d.cuh`: the N granularity is BLOCK_K, not BLOCK_N). The
/// extra row covers the second sfb row it loads when `BLOCK_K % BLOCK_N != 0`.
fn sfb_ones_len(n: usize, k: usize) -> usize {
    (n.div_ceil(128) + 1) * k.div_ceil(128)
}

pub(super) fn qwen_fp8_deepgemm_dense_enabled() -> bool {
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
pub(super) fn qwen_fp8_dense_sm_supports_deepgemm(ctx: &DeviceContext) -> bool {
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

/// Which weight-scale layout the DeepGEMM dense arm is serving. DeepGEMM reads
/// `sfb` as a `[ceil(n/128), ceil(k/128)]` f32 grid, which a per-channel `[n]`
/// scale is not, so the two layouts hand the kernel different buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fp8DeepGemmLayout {
    /// 128x128 block scales: `scale_f32` already IS the grid DeepGEMM wants.
    Blocked,
    /// Per-channel `[n]` scales: `sfb` is all ones and the channel scale goes on
    /// the output columns afterwards, because a per-output-channel scale
    /// commutes with the contraction — `sum_k (a*sa)(w*sw) = sa*sw*sum_k a*w`.
    /// Marlin's own kFE4M3fn arm factors it the same way (it scales the FP32
    /// accumulator after the K loop).
    ///
    /// This arm is NOT a re-association of the Marlin arm's arithmetic; it is a
    /// different precision. `dsv4_deepgemm_fp8_gemm_nt` has no BF16-activation
    /// entry, so `dsv4_deepgemm_pack_quantize_bf16_to_fp8` first rounds the
    /// activation to E4M3 with per-128-K-block scales: W8A16 becomes W8A8, and
    /// the activation loses 4 of its 8 mantissa bits. That dominates. The
    /// factorization itself is exact and the bf16 epilogue re-round adds one
    /// more ulp (2^-9) Marlin does not pay, but neither is the size of the
    /// error. Prefill and decode therefore compute the same projection at
    /// different precisions, and every gate on this arm has to be an
    /// end-to-end quality gate, not a parity check against Marlin.
    PerChannel,
}

/// Per-channel `[n]` FP8 scales on a shape DeepGEMM's dense NT entry accepts
/// (`k % 128`, `n % 8`). `block_k >= cols` rather than `== cols` so a TP shard,
/// whose `cols` is a slice of the K the scale was defined over, still matches —
/// the same test `repack_for_marlin_fp8` uses.
fn fp8_per_channel_deepgemm_shape(weight: &DeviceMatrix) -> bool {
    weight.weight_format == WeightFormat::Fp8BlockScaled
        && weight.quant_block_m == 1
        && weight.quant_block_k >= weight.cols
        && weight.rows.is_multiple_of(8)
        && weight.cols.is_multiple_of(128)
}

/// True when the per-channel FP8 DeepGEMM arm can fire for this weight at some
/// M this engine will actually present. The loader asks this once, at load, and
/// records the answer in `fp8_deepgemm_prefill`; nothing here can be re-derived
/// at GEMM time, because the answer also depends on which caller loaded the
/// weight.
pub(crate) fn fp8_deepgemm_per_channel_available(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
) -> bool {
    fp8_per_channel_deepgemm_shape(weight)
        && dense_deepgemm_prefill_floor(QWEN_FP8_DEEPGEMM_PER_CHANNEL_MIN_M).is_some()
        && qwen_fp8_dense_sm_supports_deepgemm(ctx)
        && qwen_fp8_deepgemm_dense_enabled()
}

/// The one decision point for both DeepGEMM dense arms, shared by the warm path
/// and the batch path so warm compiles the kernel the batch will launch. `None`
/// leaves the weight to the Marlin / dequant / GEMV arms below.
fn fp8_deepgemm_dense_layout(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    seq_len: usize,
) -> Option<Fp8DeepGemmLayout> {
    if weight.weight_format != WeightFormat::Fp8BlockScaled
        || weight.scale_f32.is_none()
        || !weight.rows.is_multiple_of(8)
        || !weight.cols.is_multiple_of(128)
        || !qwen_fp8_dense_sm_supports_deepgemm(ctx)
        || !qwen_fp8_deepgemm_dense_enabled()
    {
        return None;
    }
    if weight.quant_block_m == 128 && weight.quant_block_k == 128 {
        // 128x128 block-scaled weights never repack, so this arm still reads the
        // resident E4M3 bytes directly.
        return (weight.qweight_u8.is_some()
            && qwen_fp8_dense_route(ctx, seq_len, weight.rows, weight.cols)
                == qwen_fp8_dense_policy::Route::PackDeepGemm)
            .then_some(Fp8DeepGemmLayout::Blocked);
    }
    // Not `qwen_fp8_dense_route`: its policy is `m >= 2`, which would hand
    // DeepGEMM every batched decode step. See the floor's own derivation.
    let floor = dense_deepgemm_prefill_floor(QWEN_FP8_DEEPGEMM_PER_CHANNEL_MIN_M)?;
    // `fp8_deepgemm_prefill`, not the shape alone: the output head repacks like
    // every other dense projection but must stay on Marlin at every M.
    // B comes from the Marlin tiles when a repack ran and released the source,
    // and from the source itself when the repack declined the shape — the two
    // are independent, so a declined repack must not cost the arm.
    (weight.fp8_deepgemm_prefill
        && (weight.marlin_packed.is_some() || weight.qweight_u8.is_some())
        && fp8_per_channel_deepgemm_shape(weight)
        && seq_len >= floor)
        .then_some(Fp8DeepGemmLayout::PerChannel)
}

pub(crate) fn warm_fp8_deepgemm_dense(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    seq_len: usize,
) -> Result<bool> {
    let Some(layout) = fp8_deepgemm_dense_layout(ctx, weight, seq_len) else {
        return Ok(false);
    };
    // Warm builds the JIT kernel and discards the output, so B only has to be
    // `n * k` live bytes — the per-channel arm's Marlin tiles are exactly that,
    // and so is the E4M3 source a declined repack left resident.
    let b = match layout {
        Fp8DeepGemmLayout::Blocked => weight
            .qweight_u8
            .as_ref()
            .ok_or_else(|| anyhow!("fp8_block_scaled missing qweight_u8"))?,
        Fp8DeepGemmLayout::PerChannel => weight
            .marlin_packed
            .as_ref()
            .or(weight.qweight_u8.as_ref())
            .ok_or_else(|| anyhow!("fp8_block_scaled missing marlin_packed and qweight_u8"))?,
    };
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
    let ones = match layout {
        Fp8DeepGemmLayout::Blocked => None,
        Fp8DeepGemmLayout::PerChannel => Some(
            ctx.stream
                .clone_htod(&vec![1.0f32; sfb_ones_len(n, k)])
                .map_err(|e| anyhow!("Qwen FP8 dense DeepGEMM warm sfb ones H2D failed: {e}"))?,
        ),
    };
    let sfb = ones.as_ref().unwrap_or(scales);
    // No channel post-scale here: warm exists to build the JIT kernel and its
    // output is discarded.
    // SAFETY: ptrs from live device allocations sized to the dims passed.
    qwen_quant_profile(ctx, "qwen/fp8/dense_deepgemm_warm", m, n, k, || unsafe {
        cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
            cache_ptr(&input_fp8, ctx),
            cache_ptr(&input_scales, ctx),
            cache_ptr(b, ctx),
            cache_ptr(sfb, ctx),
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

/// Quantise the activation to E4M3 and run DeepGEMM's dense NT GEMM. Shared by
/// both prefill arms — they differ only in where `b` and `sfb` come from, and
/// the packing contract (`scale_stride_m`, the active_* triple) has to move in
/// lockstep with the kernel, so it lives in one place.
pub(super) fn deepgemm_dense_nt(
    ctx: &DeviceContext,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    b: RawDevicePtr<u8>,
    sfb: RawDevicePtr<f32>,
    m: usize,
    n: usize,
    k: usize,
    tag: &'static str,
) -> Result<()> {
    let scale_stride_m = m.div_ceil(4) * 4;
    let scale_cols = k.div_ceil(128);
    QWEN_FP8_DENSE_SCRATCH.with(|cell| -> Result<()> {
        let mut scratch = cell.borrow_mut();
        scratch.ensure(ctx, m * k, scale_stride_m * scale_cols)?;
        {
            let active_counts = scratch
                .active_counts
                .as_mut()
                .ok_or_else(|| anyhow!("{tag} DeepGEMM active_counts missing"))?;
            ctx.stream
                .memcpy_htod(&[i32::try_from(m)?], active_counts)
                .map_err(|e| anyhow!("{tag} DeepGEMM active_counts H2D failed: {e}"))?;
        }
        let input_fp8 = scratch
            .input_fp8
            .as_ref()
            .ok_or_else(|| anyhow!("{tag} DeepGEMM input scratch missing"))?;
        let input_scales = scratch
            .input_scales
            .as_ref()
            .ok_or_else(|| anyhow!("{tag} DeepGEMM scale scratch missing"))?;
        let active_experts = scratch
            .active_experts
            .as_ref()
            .ok_or_else(|| anyhow!("{tag} DeepGEMM active_experts missing"))?;
        let active_offsets = scratch
            .active_offsets
            .as_ref()
            .ok_or_else(|| anyhow!("{tag} DeepGEMM active_offsets missing"))?;
        let active_counts = scratch
            .active_counts
            .as_ref()
            .ok_or_else(|| anyhow!("{tag} DeepGEMM active_counts missing"))?;
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        qwen_quant_profile(
            ctx,
            "qwen/deepgemm/dense_pack_quantize",
            m,
            n,
            k,
            || unsafe {
                cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                    cache_ptr(input, ctx),
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
            },
        )?;
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        qwen_quant_profile(ctx, "qwen/deepgemm/dense_gemm", m, n, k, || unsafe {
            cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
                cache_ptr(input_fp8, ctx),
                cache_ptr(input_scales, ctx),
                b,
                sfb,
                cache_ptr(output, ctx),
                m,
                n,
                k,
                scale_stride_m,
                ctx.stream.cu_stream(),
            )
        })
    })
}

/// The DeepGEMM prefill arm for both FP8 layouts. `None` means the weight is
/// left to the Marlin / dequant / GEMV arms; a declined scratch reservation is
/// the same fallthrough, not an error.
fn try_fp8_deepgemm_dense(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<Option<Fp8DeepGemmLayout>> {
    let Some(layout) = fp8_deepgemm_dense_layout(ctx, weight, m) else {
        return Ok(None);
    };
    let scales = weight
        .scale_f32
        .as_ref()
        .ok_or_else(|| anyhow!("fp8_block_scaled missing scale_f32"))?;
    let n = weight.rows;
    let k = weight.cols;
    // A block-scaled weight never repacks and still holds its E4M3 bytes, and so
    // does a per-channel one whose repack declined the shape; both feed DeepGEMM
    // directly. A per-channel weight that DID repack kept only its Marlin tiles,
    // so B is un-permuted back out of them into the shared scratch —
    // byte-identical to the source it replaced.
    let b = match (layout, weight.marlin_packed.as_ref()) {
        (Fp8DeepGemmLayout::Blocked, _) | (Fp8DeepGemmLayout::PerChannel, None) => cache_ptr(
            weight
                .qweight_u8
                .as_ref()
                .ok_or_else(|| anyhow!("fp8_block_scaled missing qweight_u8"))?,
            ctx,
        ),
        (Fp8DeepGemmLayout::PerChannel, Some(packed)) => {
            let Some(ptr) =
                with_e4m3_weight_scratch(ctx, n * k, |dst| -> Result<RawDevicePtr<u8>> {
                    let dst_ptr = cache_ptr(dst, ctx);
                    qwen_quant_profile(ctx, "qwen/fp8/dense_materialize", m, n, k, || {
                        // SAFETY: ptrs from live device allocations sized to the dims passed.
                        unsafe {
                            ffi::marlin_fp8_to_e4m3_cuda(
                                cache_ptr(packed, ctx).as_ptr(),
                                dst_ptr.as_mut_ptr(),
                                n as i32,
                                k as i32,
                                ctx.stream.cu_stream(),
                            )
                            .result()
                            .map_err(|e| anyhow!("FP8 Marlin materialise kernel failed: {e}"))
                        }
                    })?;
                    Ok(dst_ptr)
                })?
            else {
                return Ok(None);
            };
            ptr
        }
    };
    let sfb = match layout {
        Fp8DeepGemmLayout::Blocked => cache_ptr(scales, ctx),
        // Per-channel scales are not DeepGEMM's [ceil(n/128), ceil(k/128)] grid,
        // so it contracts against ones and the channel scale lands in the epilogue.
        Fp8DeepGemmLayout::PerChannel => {
            QWEN_FP8_DENSE_SCRATCH.with(|cell| -> Result<RawDevicePtr<f32>> {
                let mut scratch = cell.borrow_mut();
                scratch.ensure_sfb_ones(ctx, n, k)?;
                let ones = scratch
                    .sfb_ones
                    .as_ref()
                    .ok_or_else(|| anyhow!("Qwen FP8 dense DeepGEMM sfb ones scratch missing"))?;
                Ok(cache_ptr(ones, ctx))
            })?
        }
    };
    deepgemm_dense_nt(ctx, input, output, b, sfb, m, n, k, "Qwen FP8 dense")?;
    if layout == Fp8DeepGemmLayout::PerChannel {
        let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
        let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
        qwen_quant_profile(ctx, "qwen/fp8/dense_channel_scale", m, n, k, || {
            // SAFETY: out is the [m, n] bf16 the GEMM just wrote; scales is the
            // weight's [n] channel vector (quant_block_m == 1).
            unsafe {
                ffi::scale_columns_bf16_cuda(
                    out_ptr as *mut ffi::Half,
                    scales_ptr as *const f32,
                    m as i32,
                    n as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("FP8 per-channel DeepGEMM column scale failed: {e}"))
            }
        })?;
    }
    Ok(Some(layout))
}

/// Grow the shared E4M3 weight scratch to `bytes`, returning false when the
/// allocation fails — the caller then leaves the weight to Marlin.
fn reserve_fp8_weight_scratch(ctx: &DeviceContext, bytes: usize) -> bool {
    QWEN_FP8_DEQUANT_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.fp8_capacity >= bytes {
            return true;
        }
        match ctx.stream.alloc_zeros::<u8>(bytes) {
            Ok(buf) => {
                scratch.weight_fp8 = Some(buf);
                scratch.fp8_capacity = bytes;
                true
            }
            Err(_) => false,
        }
    })
}

/// Grow the shared BF16 dequant scratch to `elems`, returning false when the
/// device is out of memory. A failed reservation is not fatal: the callers all
/// have a scalar GEMV path that computes the same result without the resident
/// BF16 copy, which is what a VRAM-committed box (full KV + recurrent pools)
/// needs to fall back to.
fn reserve_dequant_scratch(ctx: &DeviceContext, elems: usize) -> bool {
    QWEN_FP8_DEQUANT_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.capacity >= elems {
            return true;
        }
        match ctx.stream.alloc_zeros::<bf16>(elems) {
            Ok(buf) => {
                scratch.weight_bf16 = Some(buf);
                scratch.capacity = elems;
                true
            }
            Err(_) => false,
        }
    })
}

/// Reserve the shared E4M3 weight scratch and borrow it for one materialisation.
/// `None` is a declined reservation — the caller falls through to Marlin.
pub(super) fn with_e4m3_weight_scratch<T>(
    ctx: &DeviceContext,
    bytes: usize,
    f: impl FnOnce(&CudaSlice<u8>) -> Result<T>,
) -> Result<Option<T>> {
    if !reserve_fp8_weight_scratch(ctx, bytes) {
        return Ok(None);
    }
    QWEN_FP8_DEQUANT_SCRATCH.with(|cell| {
        let scratch = cell.borrow();
        // reserve succeeded => weight_fp8 is Some.
        let dst = scratch.weight_fp8.as_ref().expect("reserved above");
        f(dst).map(Some)
    })
}

/// Reserve the shared BF16 dequant scratch and borrow it for one dequant+GEMM.
/// `None` is a declined reservation — the caller falls through to the GEMV.
pub(super) fn with_dequant_weight_scratch<T>(
    ctx: &DeviceContext,
    elems: usize,
    f: impl FnOnce(&CudaSlice<bf16>) -> Result<T>,
) -> Result<Option<T>> {
    if !reserve_dequant_scratch(ctx, elems) {
        return Ok(None);
    }
    QWEN_FP8_DEQUANT_SCRATCH.with(|cell| {
        let scratch = cell.borrow_mut();
        // reserve succeeded => weight_bf16 is Some.
        let weight_bf16 = scratch.weight_bf16.as_ref().expect("reserved above");
        f(weight_bf16).map(Some)
    })
}

/// Pre-Hopper dense FP8 GEMM fallback for the DeepGEMM-shaped path: on sm < 9
/// (A100/V100), where DeepGEMM's `wgmma` kernel is unavailable, dequantize the
/// FP8 E4M3 block-scaled weight `[rows, cols]` to a resident BF16 scratch ONCE,
/// then run the existing BF16 cuBLAS GEMM (`gemm_cuda`). This is the large-M
/// (prefill) path: dequant is `rows*cols` work amortized over the `M` GEMM rows,
/// far cheaper than the warp-per-row scalar GEMV at large M.
///
/// Engages for large-M (prefill) on ANY arch whenever DeepGEMM did not handle the
/// GEMM — we only reach this fn *after* `try_fp8_deepgemm_dense` returned
/// false (DeepGEMM disabled/unbuilt, or it declined the shape). Prefill must NEVER
/// fall through to the scalar GEMV below — that is a memory-bound per-token path
/// (~20× slower at M=2048): dequant once, cuBLAS GEMM over all M rows instead.
///
/// Only engages when the weight is `Fp8BlockScaled` (any block shape — per-channel
/// `[1, K]` included, since DeepGEMM above takes those only from
/// [`dense_deepgemm_prefill_floor`] and only on Hopper) and [`fp8_route`] says
/// `DequantGemm` or `DeepGemm` declined — i.e. at or above
/// `QWEN_DEQUANT_GEMM_PREFILL_MIN_M` on a weight the Marlin repack declined.
///
/// The M floor is load-bearing. The dequant is per call, not cached, so at
/// decode M this path re-materialises every FP8 weight in the model on every
/// step: on Qwen3.8-27B-NVFP4 that is 11.56 G params, +85 ms/step, measured as
/// a 5× aggregate throughput loss at c=16.
///
/// Returns `Ok(false)` when it does not apply, leaving the route to the scalar
/// GEMV.
fn try_fp8_dequant_bf16_gemm(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<bool> {
    if weight.weight_format != WeightFormat::Fp8BlockScaled {
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

    let Some(()) = with_dequant_weight_scratch(ctx, weight_elems, |weight_bf16| -> Result<()> {
        let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
        let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
        let (wbf16_ptr, _gw) = weight_bf16.device_ptr(&ctx.stream);
        let stream = ctx.stream.cu_stream();
        qwen_quant_profile(
            ctx,
            "qwen/fp8/dense_dequant_bf16",
            m,
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
        let (x_ptr, _gx) = input.device_ptr(&ctx.stream);
        let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
        qwen_quant_profile(
            ctx,
            "qwen/fp8/dense_dequant_gemm",
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
                .map_err(|e| anyhow!("Qwen FP8 dense dequant BF16 GEMM failed: {e}"))
            },
        )?;
        Ok(())
    })?
    else {
        return Ok(false);
    };
    Ok(true)
}

/// The one place that decides how an FP8 weight DeepGEMM declined is multiplied.
/// Mirrors NVFP4's single-path shape, with one difference that is load-bearing.
///
/// - `Marlin` (tensor core) at every M the DeepGEMM arm above did not claim —
///   which is every M for a per-channel weight off Hopper, and everything below
///   [`dense_deepgemm_prefill_floor`] on it. Both single-row lanes
///   (`gemv`, and `gemm_batch` at m=1 from the MTP draft head) land here.
/// - `DequantGemm` for weights with NO Marlin layout, but only
///   from [`QWEN_DEQUANT_GEMM_PREFILL_MIN_M`], not from the M=2 floor FP4 uses.
///   An un-repacked FP8 weight is a 128×128 block weight on a non-Hopper card,
///   and dequantising all of it at M=2 is the defect in
///   `docs/experience/errors/2026-08-19-fp8-dequant-arm-shadows-decode.md`.
/// - `Gemv` (batched scalar warp-per-row) below that.
///
/// `DeepGemm` is the query-level prefill verdict; the arm above stays the sole
/// eligibility gate (its route policy and scratch reservation can decline
/// dynamically), and a declined DeepGemm falls through to the dequant/GEMV
/// order this fn gives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Fp8Route {
    DeepGemm,
    Marlin,
    DequantGemm,
    Gemv,
}

#[derive(Clone, Copy)]
pub(super) struct Fp8Query {
    pub(super) repacked: bool,
    pub(super) source: bool,
    pub(super) per_channel_shape: bool,
    pub(super) deepgemm_prefill: bool,
}

pub(super) fn fp8_route(q: Fp8Query, m: usize, sm_marlin: bool, deepgemm_on: bool) -> Fp8Route {
    if q.repacked && sm_marlin {
        return Fp8Route::Marlin;
    }
    if deepgemm_on
        && q.deepgemm_prefill
        && q.per_channel_shape
        && m >= QWEN_FP8_DEEPGEMM_PER_CHANNEL_MIN_M
    {
        return Fp8Route::DeepGemm;
    }
    if q.source && m >= QWEN_DEQUANT_GEMM_PREFILL_MIN_M {
        return Fp8Route::DequantGemm;
    }
    Fp8Route::Gemv
}

/// Per-channel FP8 Marlin tensor-core GEMM (Ampere+): C[m,n] = X[m,k] @
/// dequant(W). Fires when the SM gate is on, the weight was Marlin-repacked at
/// load (`repack_for_marlin_fp8`, per-channel only), and M is inside the window
/// where Marlin beats dequant→cuBLAS. Supersedes the batched FP8 GEMV there —
/// that kernel's tile is the batch, so it re-reads the weight above B=8, which
/// is the whole NVFP4-vs-FP8 concurrency gap.
fn marlin_fp8_gemm_raw(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<bool> {
    let (Some(packed), Some(scales)) =
        (weight.marlin_packed.as_ref(), weight.marlin_scales.as_ref())
    else {
        unreachable!("fp8_route returns Marlin only when both are Some")
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
        qwen_quant_profile(ctx, "qwen/fp8/marlin_gemm", m, n, k, || {
            // SAFETY: all ptrs from live device allocations; packed/scales sized by
            // repack_for_marlin_fp8 for these dims, x=[m,k], out=[m,n],
            // c_tmp/workspace sized to the SM max.
            unsafe {
                ffi::marlin_fp8_gemm_cuda(
                    x_ptr as *const ffi::Half,
                    packed_ptr as *const u32,
                    scales_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    c_tmp_ptr as *mut f32,
                    ws_ptr as *mut i32,
                    m as i32,
                    n as i32,
                    k as i32,
                    stream,
                )
                .result()
                .map_err(|e| anyhow!("FP8 per-channel Marlin GEMM failed: {e}"))
            }
        })?;
        Ok(())
    })?;
    MARLIN_FP8_HITS.fetch_add(1, Ordering::Relaxed);
    Ok(true)
}

/// Scalar/batched block-scaled GEMV — the terminal FP8 fallback. M=1 takes the
/// singular ABI the decode lane always used; larger M takes the batched ABI.
fn fp8_block_scaled_gemv(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<()> {
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
    let (x_ptr, _gx) = input.device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();
    if m == 1 {
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
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
            .result()?
        };
    } else {
        // The coalesced scalar warp-per-row GEMV is the production path:
        // 3.6x faster than the tensor-core MMA tile at B=1 decode on H20
        // (the MMA was occupancy-starved + uncoalesced off its batched
        // B<=16 design point — KILLED, see
        // docs/experience/wins/2026-06-22-qwen-fp8-decode-gemv-scalar.md).
        qwen_quant_profile(
            ctx,
            "qwen/fp8/gemv_batch",
            m,
            weight.rows,
            weight.cols,
            || {
                Ok(unsafe {
                    ffi::gemv_fp8_block_scaled_batch_cuda(
                        qw_ptr as *const u8,
                        scales_ptr as *const f32,
                        x_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        m as i32,
                        weight.rows as i32,
                        weight.cols as i32,
                        scale_rows,
                        scale_cols,
                        block_m,
                        block_k,
                        stream,
                    )
                    .result()?
                })
            },
        )?;
    }
    Ok(())
}

pub(super) fn run(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<()> {
    // DeepGEMM stays first unconditionally: its internal gates (route policy,
    // scratch reservation) can decline, and a decline must fall through.
    if let Some(layout) = try_fp8_deepgemm_dense(ctx, weight, input, output, m)? {
        let counter = match layout {
            Fp8DeepGemmLayout::Blocked => &DEEPGEMM_HITS,
            Fp8DeepGemmLayout::PerChannel => &DEEPGEMM_PER_CHANNEL_HITS,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        return Ok(());
    }
    let query = Fp8Query {
        repacked: weight.marlin_packed.is_some() && weight.marlin_scales.is_some(),
        source: weight.qweight_u8.is_some(),
        per_channel_shape: fp8_per_channel_deepgemm_shape(weight),
        deepgemm_prefill: weight.fp8_deepgemm_prefill,
    };
    match fp8_route(
        query,
        m,
        marlin_sm_supported(ctx),
        qwen_fp8_deepgemm_dense_enabled(),
    ) {
        Fp8Route::DeepGemm | Fp8Route::DequantGemm => {
            if try_fp8_dequant_bf16_gemm(ctx, weight, input, output, m)? {
                FP8_DEQUANT_GEMM_HITS.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
            fp8_block_scaled_gemv(ctx, weight, input, output, m)?;
            FP8_GEMV_HITS.fetch_add(1, Ordering::Relaxed);
        }
        Fp8Route::Marlin => {
            if !marlin_fp8_gemm_raw(ctx, weight, input, output, m)? {
                fp8_block_scaled_gemv(ctx, weight, input, output, m)?;
                FP8_GEMV_HITS.fetch_add(1, Ordering::Relaxed);
            }
        }
        Fp8Route::Gemv => {
            fp8_block_scaled_gemv(ctx, weight, input, output, m)?;
            FP8_GEMV_HITS.fetch_add(1, Ordering::Relaxed);
        }
    }
    Ok(())
}
