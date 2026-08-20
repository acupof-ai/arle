use anyhow::{Result, anyhow, ensure};
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix};
use cuda_kernels::quant_linear as cuda_ql;
use cuda_kernels::tensor::{WeightFormat, cache_ptr};
use cudarc::driver::CudaSlice;
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
    with_marlin_scratch(ctx, |scratch| {
        let c_tmp = scratch.c_tmp.as_ref().unwrap();
        let workspace = scratch.workspace.as_ref().unwrap();
        qwen_quant_profile(ctx, "qwen/w8a16/marlin_gemm", m, n, k, || {
            cuda_ql::marlin_w8a16_gemm(
                ctx,
                input,
                packed,
                scales,
                output,
                c_tmp,
                workspace,
                m,
                n,
                k,
                weight.group_size,
            )
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
        let wbf16 = cache_ptr(weight_bf16, ctx);
        // SAFETY: the scratch covers `weight_elems` and lives across both launches.
        qwen_quant_profile(ctx, "qwen/w8a16/dense_dequant_bf16", m, n, k, || unsafe {
            cuda_ql::dequantize_w8a16_to_bf16(ctx, qw, scales, wbf16, n, k, weight.group_size)
        })?;
        // SAFETY: same scratch, fully written by the dequant above.
        qwen_quant_profile(ctx, "qwen/w8a16/dense_dequant_gemm", m, n, k, || unsafe {
            cuda_ql::gemm_bf16(ctx, wbf16, input, output, m, n, k)
        })?;
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
    if m == 1 {
        cuda_ql::w8a16_gemv(
            ctx,
            qw,
            scales,
            input,
            output,
            weight.rows,
            weight.cols,
            weight.group_size,
        )?;
    } else {
        cuda_ql::w8a16_gemv_batch(
            ctx,
            qw,
            scales,
            input,
            output,
            m,
            weight.rows,
            weight.cols,
            weight.group_size,
        )?;
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
    if m == 1 {
        cuda_ql::w4a16_gemv(
            ctx,
            qw,
            scales,
            input,
            output,
            weight.rows,
            weight.cols,
            weight.group_size,
        )?;
    } else {
        cuda_ql::w4a16_gemv_batch(
            ctx,
            qw,
            scales,
            input,
            output,
            m,
            weight.rows,
            weight.cols,
            weight.group_size,
        )?;
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

/// The INT storage states the load validator inspects. The W8A16 repack frees
/// its source pair, so post-repack the Marlin pair must be complete — the
/// historical W8A16 lm_head defect was exactly a freed source with no consumer
/// on one lane.
#[derive(Clone, Copy)]
pub(super) struct IntStorage {
    pub(super) marlin_packed: bool,
    pub(super) marlin_scales: bool,
    pub(super) source_weight: bool,
    pub(super) source_scales: bool,
    pub(super) is_w8a16: bool,
    pub(super) group_aligned: bool,
}

impl IntStorage {
    pub(super) fn of(weight: &DeviceMatrix) -> Self {
        Self {
            marlin_packed: weight.marlin_packed.is_some(),
            marlin_scales: weight.marlin_scales.is_some(),
            source_weight: weight.qweight.is_some(),
            source_scales: weight.qscales.is_some(),
            is_w8a16: weight.weight_format == WeightFormat::W8A16,
            group_aligned: weight.group_size > 0 && weight.cols.is_multiple_of(weight.group_size),
        }
    }
}

/// `None` when every M [`int_route`] can produce has a resident consumer.
pub(super) fn int_missing_representation(s: IntStorage, sm_marlin: bool) -> Option<&'static str> {
    if s.marlin_packed != s.marlin_scales {
        return Some("the other half of the Marlin pair (marlin_packed XOR marlin_scales)");
    }
    if s.source_weight != s.source_scales {
        return Some("the other half of the source pair (qweight XOR qscales)");
    }
    let source = s.source_weight && s.source_scales;
    if !s.is_w8a16 {
        // W4A16: the group-wise GEMV source pair is its only route.
        if !source {
            return Some("the qweight + qscales source pair (W4A16's only route)");
        }
        return (!s.group_aligned).then_some("a group size > 0 dividing cols (W4A16 GEMV)");
    }
    if source && !s.group_aligned {
        return Some("a group size > 0 dividing cols (W8A16 GEMV/dequant)");
    }
    let marlin = s.marlin_packed && s.marlin_scales && sm_marlin;
    (!marlin && !source)
        .then_some("a retained Marlin pair (sm_80+) or the qweight + qscales source pair")
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
